import * as tf from '@tensorflow/tfjs';
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

const DATA_DIR = resolve('../data/co-data');
const OUTPUT_DIR = resolve('../artifacts/json_outputs');
const TIME_STEPS = 30;
const FORECAST_HORIZON_STEPS = 30;
const TEST_SPLIT = 0.15;
const EPOCHS = 180;
const BATCH_SIZE = 32;
const MIN_TEST_TARGET_CO_PPM = 35;
const MAX_CO_PPM = 200;

const parseCsv = (path) => {
  const [headerLine, ...lines] = readFileSync(path, 'utf8').trim().split(/\r?\n/);
  const headers = headerLine.split(',').map((value) => value.trim());
  const normalizedHeaders = headers.map((header) => header.toLowerCase().replace(/[^a-z0-9]/g, ''));
  const coIndex = normalizedHeaders.findIndex((header) => header === 'coppm');
  const timeIndex = normalizedHeaders.findIndex((header) => header === 'timestamps' || header === 'timestampms');
  const timestampIsMs = normalizedHeaders[timeIndex] === 'timestampms';
  if (coIndex < 0) {
    return [];
  }

  return lines.map((line, rowIndex) => {
    const cols = line.split(',').map((value) => value.trim());
    const headerAlignedCo = Number(cols[coIndex]);
    // Some new_sensor_data rows omit Altitude(m), which shifts MQ7/Voltage/CO left.
    // In those rows, CO(ppm) is still the final column.
    const coPpm = Number.isFinite(headerAlignedCo) ? headerAlignedCo : Number(cols.at(-1));
    const rawTimestamp = timeIndex >= 0 ? Number(cols[timeIndex]) : rowIndex * 2;
    return {
      timestampSeconds: timestampIsMs ? rawTimestamp / 1000 : rawTimestamp,
      // Preserve the mobile GRU feature layout: [coPpm, bpm, accelX, accelY, accelZ].
      // These CO CSVs do not contain wearable channels, so they are zero-filled
      // exactly like the app's CSV training adapter.
      features: [coPpm, 0, 0, 0, 0],
      coPpm,
      sourceFile: path.split('/').at(-1),
    };
  }).filter((sample) => (
    Number.isFinite(sample.coPpm)
    && sample.coPpm <= MAX_CO_PPM
    && Number.isFinite(sample.timestampSeconds)
  ));
};

const computeStats = (windows) => {
  const featureCount = windows[0][0].length;
  const mean = Array(featureCount).fill(0);
  const std = Array(featureCount).fill(0);
  let count = 0;

  for (const window of windows) {
    for (const row of window) {
      for (let i = 0; i < featureCount; i += 1) {
        mean[i] += row[i];
      }
      count += 1;
    }
  }
  for (let i = 0; i < featureCount; i += 1) {
    mean[i] /= count;
  }
  for (const window of windows) {
    for (const row of window) {
      for (let i = 0; i < featureCount; i += 1) {
        std[i] += (row[i] - mean[i]) ** 2;
      }
    }
  }
  for (let i = 0; i < featureCount; i += 1) {
    std[i] = Math.sqrt(std[i] / count) || 1;
  }
  return { mean, std };
};

const normalizeWindows = (windows, stats) => windows.map((window) => (
  window.map((row) => row.map((value, index) => (value - stats.mean[index]) / stats.std[index]))
));

const buildSequences = (sampleGroups) => {
  const sequences = [];
  const sequenceCountsByFile = {};
  const eligibleCountsByFile = {};
  const rawSampleCountsByFile = {};

  for (const [sourceFile, samples] of sampleGroups) {
    rawSampleCountsByFile[sourceFile] = samples.length;
    sequenceCountsByFile[sourceFile] = 0;
    eligibleCountsByFile[sourceFile] = 0;
    if (samples.length <= TIME_STEPS + FORECAST_HORIZON_STEPS) {
      continue;
    }

    for (let start = 0; start + TIME_STEPS + FORECAST_HORIZON_STEPS < samples.length; start += 1) {
      const target = samples[start + TIME_STEPS + FORECAST_HORIZON_STEPS];
      sequenceCountsByFile[sourceFile] += 1;
      if (target.coPpm < MIN_TEST_TARGET_CO_PPM || target.coPpm > MAX_CO_PPM) {
        continue;
      }

      eligibleCountsByFile[sourceFile] += 1;
      sequences.push({
        window: samples.slice(start, start + TIME_STEPS).map((sample) => sample.features),
        target: target.coPpm,
        targetTime: target.timestampSeconds,
        sourceFile,
      });
    }
  }

  return {
    sequences,
    rawSampleCountsByFile,
    sequenceCountsByFile,
    eligibleCountsByFile,
  };
};

const readSampleGroups = () => {
  const csvFiles = readdirSync(DATA_DIR)
    .filter((fileName) => fileName.toLowerCase().endsWith('.csv'))
    .sort();
  return csvFiles.map((fileName) => [fileName, parseCsv(resolve(DATA_DIR, fileName))]);
};

const unzipSequences = (sequences) => {
  const windows = [];
  const targets = [];
  const targetTimes = [];
  const sourceFiles = [];
  for (const sequence of sequences) {
    windows.push(sequence.window);
    targets.push(sequence.target);
    targetTimes.push(sequence.targetTime);
    sourceFiles.push(sequence.sourceFile);
  }
  return { windows, targets, targetTimes, sourceFiles };
};

const buildModel = () => {
  const model = tf.sequential();
  model.add(tf.layers.gru({
    inputShape: [TIME_STEPS, 5],
    units: 48,
    returnSequences: false,
    recurrentInitializer: 'glorotUniform',
  }));
  model.add(tf.layers.dropout({ rate: 0.15 }));
  model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae'],
  });
  return model;
};

const main = async () => {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  await tf.ready();

  const sampleGroups = readSampleGroups();
  const {
    sequences,
    rawSampleCountsByFile,
    sequenceCountsByFile,
    eligibleCountsByFile,
  } = buildSequences(sampleGroups);
  if (sequences.length < 2) {
    throw new Error(`Not enough ${MIN_TEST_TARGET_CO_PPM}+ ppm forecasting windows were found in ${DATA_DIR}`);
  }

  const { windows, targets, targetTimes, sourceFiles } = unzipSequences(sequences);
  const testCount = Math.max(1, Math.round(windows.length * TEST_SPLIT));
  const trainCount = windows.length - testCount;

  const trainWindows = windows.slice(0, trainCount);
  const testWindows = windows.slice(trainCount);
  const trainTargets = targets.slice(0, trainCount);
  const testTargets = targets.slice(trainCount);
  const testTargetTimes = targetTimes.slice(trainCount);
  const testSourceFiles = sourceFiles.slice(trainCount);

  const stats = computeStats(trainWindows);
  const normalizedTrain = normalizeWindows(trainWindows, stats);
  const normalizedTest = normalizeWindows(testWindows, stats);

  const xsTrain = tf.tensor3d(normalizedTrain, [trainCount, TIME_STEPS, 5], 'float32');
  const ysTrain = tf.tensor2d(trainTargets, [trainCount, 1], 'float32');
  const xsTest = tf.tensor3d(normalizedTest, [testCount, TIME_STEPS, 5], 'float32');
  const ysTest = tf.tensor2d(testTargets, [testCount, 1], 'float32');
  const model = buildModel();

  const history = await model.fit(xsTrain, ysTrain, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [xsTest, ysTest],
    shuffle: false,
    verbose: 0,
  });

  const predictions = Array.from(model.predict(xsTest).dataSync());
  const mae = predictions.reduce((sum, prediction, index) => (
    sum + Math.abs(prediction - testTargets[index])
  ), 0) / predictions.length;
  const rmse = Math.sqrt(predictions.reduce((sum, prediction, index) => (
    sum + (prediction - testTargets[index]) ** 2
  ), 0) / predictions.length);

  const rows = predictions.map((prediction, index) => ({
    timestampSeconds: testTargetTimes[index],
    sourceFile: testSourceFiles[index],
    actualCoPpm: testTargets[index],
    predictedCoPpm: prediction,
  }));
  const summary = {
    generatedAt: new Date().toISOString(),
    framework: 'TensorFlow.js',
    modelType: 'GRU',
    dataDir: DATA_DIR,
    sourceFiles: sampleGroups.map(([sourceFile]) => sourceFile),
    rawSampleCountsByFile,
    sequenceCountsByFile,
    eligibleTargetMinimumCoPpm: MIN_TEST_TARGET_CO_PPM,
    maximumCoPpmIncluded: MAX_CO_PPM,
    eligibleSequenceCountsByFile: eligibleCountsByFile,
    eligibleSequenceCount: sequences.length,
    sequenceCount: windows.length,
    trainSequenceCount: trainCount,
    testSequenceCount: testCount,
    testSplit: TEST_SPLIT,
    inputWindowLengthTimesteps: TIME_STEPS,
    forecastHorizonSteps: FORECAST_HORIZON_STEPS,
    inputFeatures: ['coPpm', 'bpm_zero_filled', 'accelX_zero_filled', 'accelY_zero_filled', 'accelZ_zero_filled'],
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    finalTrainLoss: history.history.loss.at(-1),
    finalValidationLoss: history.history.val_loss.at(-1),
    finalTrainMae: history.history.mae.at(-1),
    finalValidationMae: history.history.val_mae.at(-1),
    testMae: mae,
    testRmse: rmse,
    featureStats: stats,
    predictions: rows,
  };

  writeFileSync(resolve(OUTPUT_DIR, 'gru_new_sensor_predictions.json'), JSON.stringify(summary, null, 2));
  writeFileSync(
    resolve(OUTPUT_DIR, 'gru_new_sensor_predictions.csv'),
    [
      'sourceFile,timestampSeconds,actualCoPpm,predictedCoPpm',
      ...rows.map((row) => `${row.sourceFile},${row.timestampSeconds},${row.actualCoPpm},${row.predictedCoPpm}`),
    ].join('\n'),
  );

  xsTrain.dispose();
  ysTrain.dispose();
  xsTest.dispose();
  ysTest.dispose();
  model.dispose();

  console.log(JSON.stringify({
    sourceFiles: summary.sourceFiles,
    eligibleTargetMinimumCoPpm: MIN_TEST_TARGET_CO_PPM,
    eligibleSequenceCountsByFile: eligibleCountsByFile,
    sequenceCount: windows.length,
    trainSequenceCount: trainCount,
    testSequenceCount: testCount,
    testMae: mae,
    testRmse: rmse,
    finalValidationMae: summary.finalValidationMae,
  }, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
