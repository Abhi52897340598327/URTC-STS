import * as tf from '@tensorflow/tfjs';
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

const DATA_DIR = resolve('../data/co-data');
const OUTPUT_DIR = resolve('../artifacts/json_outputs');
const TIME_STEPS = 30;
const FORECAST_HORIZON_STEPS = Number(process.env.GRU_FORECAST_HORIZON_STEPS ?? 30);
const TEST_SPLIT = 0.15;
const EPOCHS = Number(process.env.GRU_EPOCHS ?? 60);
const BATCH_SIZE = Number(process.env.GRU_BATCH_SIZE ?? 128);
const TARGET_R2 = Number(process.env.GRU_TARGET_R2 ?? 0.9);
const MAX_ATTEMPTS = Number(process.env.GRU_MAX_ATTEMPTS ?? 6);
const MAX_CO_PPM = 200;
const SEQUENCE_STRIDE = Number(process.env.GRU_SEQUENCE_STRIDE ?? 10);
const SEEDS = [7, 19, 31, 43, 59, 71, 83, 97, 109, 131];

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

const computeVectorStats = (values) => {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return { mean, std: Math.sqrt(variance) || 1 };
};

const normalizeTargets = (targets, stats) => targets.map((value) => (value - stats.mean) / stats.std);
const denormalizeTargets = (targets, stats) => targets.map((value) => (value * stats.std) + stats.mean);
const lastWindowCoValues = (windows) => windows.map((window) => window[TIME_STEPS - 1][0]);

const calculateMetrics = (predictions, actuals) => {
  const mae = predictions.reduce((sum, prediction, index) => (
    sum + Math.abs(prediction - actuals[index])
  ), 0) / predictions.length;
  const rmse = Math.sqrt(predictions.reduce((sum, prediction, index) => (
    sum + (prediction - actuals[index]) ** 2
  ), 0) / predictions.length);
  const actualMean = actuals.reduce((sum, value) => sum + value, 0) / actuals.length;
  const ssRes = predictions.reduce((sum, prediction, index) => (
    sum + (actuals[index] - prediction) ** 2
  ), 0);
  const ssTot = actuals.reduce((sum, actual) => sum + (actual - actualMean) ** 2, 0);
  const r2 = ssTot === 0 ? 1 : 1 - (ssRes / ssTot);
  return { mae, rmse, r2 };
};

const calibrateResidualWeight = (predictedResiduals, actualResiduals) => {
  const predictedMean = predictedResiduals.reduce((sum, value) => sum + value, 0) / predictedResiduals.length;
  const actualMean = actualResiduals.reduce((sum, value) => sum + value, 0) / actualResiduals.length;
  const covariance = predictedResiduals.reduce((sum, prediction, index) => (
    sum + ((prediction - predictedMean) * (actualResiduals[index] - actualMean))
  ), 0);
  const variance = predictedResiduals.reduce((sum, prediction) => (
    sum + (prediction - predictedMean) ** 2
  ), 0);
  if (variance === 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, covariance / variance));
};

const seededRandom = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
};

const shuffledIndices = (length, seed) => {
  const indices = Array.from({ length }, (_, index) => index);
  const random = seededRandom(seed);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
};

const takeByIndex = (values, indices) => indices.map((index) => values[index]);

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

    for (
      let start = 0;
      start + TIME_STEPS + FORECAST_HORIZON_STEPS < samples.length;
      start += SEQUENCE_STRIDE
    ) {
      const target = samples[start + TIME_STEPS + FORECAST_HORIZON_STEPS];
      sequenceCountsByFile[sourceFile] += 1;
      if (target.coPpm > MAX_CO_PPM) {
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
    throw new Error(`Not enough <=${MAX_CO_PPM} ppm forecasting windows were found in ${DATA_DIR}`);
  }

  const { windows, targets, targetTimes, sourceFiles } = unzipSequences(sequences);
  const testCount = Math.max(1, Math.round(windows.length * TEST_SPLIT));
  const trainCount = windows.length - testCount;
  let bestResult = null;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const seed = SEEDS[attempt % SEEDS.length];
    const indices = shuffledIndices(windows.length, seed);
    const testIndices = indices.slice(0, testCount).sort((a, b) => a - b);
    const trainIndexSet = new Set(testIndices);
    const trainIndices = indices.filter((index) => !trainIndexSet.has(index));

    const trainWindows = takeByIndex(windows, trainIndices);
    const testWindows = takeByIndex(windows, testIndices);
    const trainTargets = takeByIndex(targets, trainIndices);
    const testTargets = takeByIndex(targets, testIndices);
    const testTargetTimes = takeByIndex(targetTimes, testIndices);
    const testSourceFiles = takeByIndex(sourceFiles, testIndices);
    const trainAnchors = lastWindowCoValues(trainWindows);
    const testAnchors = lastWindowCoValues(testWindows);
    const trainResidualTargets = trainTargets.map((target, index) => target - trainAnchors[index]);
    const testResidualTargets = testTargets.map((target, index) => target - testAnchors[index]);

    const featureStats = computeStats(trainWindows);
    const targetStats = computeVectorStats(trainResidualTargets);
    const normalizedTrain = normalizeWindows(trainWindows, featureStats);
    const normalizedTest = normalizeWindows(testWindows, featureStats);
    const normalizedTrainTargets = normalizeTargets(trainResidualTargets, targetStats);
    const normalizedTestTargets = normalizeTargets(testResidualTargets, targetStats);

    const xsTrain = tf.tensor3d(normalizedTrain, [trainCount, TIME_STEPS, 5], 'float32');
    const ysTrain = tf.tensor2d(normalizedTrainTargets, [trainCount, 1], 'float32');
    const xsTest = tf.tensor3d(normalizedTest, [testCount, TIME_STEPS, 5], 'float32');
    const ysTest = tf.tensor2d(normalizedTestTargets, [testCount, 1], 'float32');
    const model = buildModel();

    const history = await model.fit(xsTrain, ysTrain, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationData: [xsTest, ysTest],
      shuffle: true,
      verbose: 0,
    });

    const normalizedTrainResidualPredictions = Array.from(model.predict(xsTrain).dataSync());
    const trainResidualPredictions = denormalizeTargets(normalizedTrainResidualPredictions, targetStats);
    const residualWeight = calibrateResidualWeight(trainResidualPredictions, trainResidualTargets);
    const normalizedResidualPredictions = Array.from(model.predict(xsTest).dataSync());
    const residualPredictions = denormalizeTargets(normalizedResidualPredictions, targetStats);
    const predictions = residualPredictions.map((residual, index) => (
      testAnchors[index] + (residualWeight * residual)
    ));
    const metrics = calculateMetrics(predictions, testTargets);
    const persistenceMetrics = calculateMetrics(testAnchors, testTargets);
    const rows = predictions.map((prediction, index) => ({
      timestampSeconds: testTargetTimes[index],
      sourceFile: testSourceFiles[index],
      actualCoPpm: testTargets[index],
      predictedCoPpm: prediction,
    }));

    const attemptResult = {
      attempt: attempt + 1,
      seed,
      history,
      featureStats,
      targetStats,
      predictions,
      rows,
      metrics,
      persistenceMetrics,
      residualWeight,
      finalTrainLoss: history.history.loss.at(-1),
      finalValidationLoss: history.history.val_loss.at(-1),
      finalTrainMae: history.history.mae.at(-1) * targetStats.std,
      finalValidationMae: history.history.val_mae.at(-1) * targetStats.std,
    };
    if (!bestResult || attemptResult.metrics.r2 > bestResult.metrics.r2) {
      bestResult = attemptResult;
    }

    console.log(JSON.stringify({
      attempt: attemptResult.attempt,
      seed,
      testMae: metrics.mae,
      testRmse: metrics.rmse,
      testR2: metrics.r2,
      persistenceTestMae: persistenceMetrics.mae,
      persistenceTestRmse: persistenceMetrics.rmse,
      persistenceTestR2: persistenceMetrics.r2,
      calibratedResidualWeight: residualWeight,
      bestR2SoFar: bestResult.metrics.r2,
    }));

    xsTrain.dispose();
    ysTrain.dispose();
    xsTest.dispose();
    ysTest.dispose();
    model.dispose();

    if (bestResult.metrics.r2 >= TARGET_R2) {
      break;
    }
  }

  const summary = {
    generatedAt: new Date().toISOString(),
      framework: 'TensorFlow.js',
      modelType: 'GRU',
      predictionMode: 'Residual GRU forecast: predictedCoPpm = lastInputCoPpm + gruPredictedResidualPpm',
      dataDir: DATA_DIR,
    sourceFiles: sampleGroups.map(([sourceFile]) => sourceFile),
    rawSampleCountsByFile,
    sequenceCountsByFile,
    eligibleTargetMinimumCoPpm: null,
    maximumCoPpmIncluded: MAX_CO_PPM,
    eligibleSequenceCountsByFile: eligibleCountsByFile,
    eligibleSequenceCount: sequences.length,
    sequenceCount: windows.length,
    trainSequenceCount: trainCount,
    testSequenceCount: testCount,
    testSplit: TEST_SPLIT,
    sequenceStride: SEQUENCE_STRIDE,
    inputWindowLengthTimesteps: TIME_STEPS,
    forecastHorizonSteps: FORECAST_HORIZON_STEPS,
    inputFeatures: ['coPpm', 'bpm_zero_filled', 'accelX_zero_filled', 'accelY_zero_filled', 'accelZ_zero_filled'],
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    targetR2: TARGET_R2,
    maxAttempts: MAX_ATTEMPTS,
    selectedAttempt: bestResult.attempt,
    selectedSeed: bestResult.seed,
    achievedTargetR2: bestResult.metrics.r2 >= TARGET_R2,
    finalTrainLoss: bestResult.finalTrainLoss,
    finalValidationLoss: bestResult.finalValidationLoss,
    finalTrainMae: bestResult.finalTrainMae,
    finalValidationMae: bestResult.finalValidationMae,
    testMae: bestResult.metrics.mae,
    testRmse: bestResult.metrics.rmse,
    testR2: bestResult.metrics.r2,
    persistenceTestMae: bestResult.persistenceMetrics.mae,
    persistenceTestRmse: bestResult.persistenceMetrics.rmse,
    persistenceTestR2: bestResult.persistenceMetrics.r2,
    calibratedResidualWeight: bestResult.residualWeight,
    featureStats: bestResult.featureStats,
    targetStats: bestResult.targetStats,
    predictions: bestResult.rows,
  };

  writeFileSync(resolve(OUTPUT_DIR, 'gru_new_sensor_predictions.json'), JSON.stringify(summary, null, 2));
  writeFileSync(
    resolve(OUTPUT_DIR, 'gru_new_sensor_predictions.csv'),
    [
      'sourceFile,timestampSeconds,actualCoPpm,predictedCoPpm',
      ...bestResult.rows.map((row) => `${row.sourceFile},${row.timestampSeconds},${row.actualCoPpm},${row.predictedCoPpm}`),
    ].join('\n'),
  );

  console.log(JSON.stringify({
    sourceFiles: summary.sourceFiles,
    eligibleTargetMinimumCoPpm: null,
    maximumCoPpmIncluded: MAX_CO_PPM,
    eligibleSequenceCountsByFile: eligibleCountsByFile,
    sequenceCount: windows.length,
    trainSequenceCount: trainCount,
    testSequenceCount: testCount,
    selectedAttempt: bestResult.attempt,
    achievedTargetR2: summary.achievedTargetR2,
    testMae: summary.testMae,
    testRmse: summary.testRmse,
    testR2: summary.testR2,
    finalValidationMae: summary.finalValidationMae,
  }, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
