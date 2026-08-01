import * as tf from '@tensorflow/tfjs';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const DATA_DIR = resolve('../data/co-data');
const OUTPUT_DIR = resolve('../artifacts/json_outputs');
const EXISTING_SUMMARY_PATH = resolve(OUTPUT_DIR, 'gru_new_sensor_predictions.json');
const TIME_STEPS = 30;
const FORECAST_HORIZON_STEPS = Number(process.env.GRU_FORECAST_HORIZON_STEPS ?? 1);
const TEST_SPLIT = 0.15;
const EPOCHS = Number(process.env.CO_COMPARE_EPOCHS ?? 20);
const BATCH_SIZE = Number(process.env.CO_COMPARE_BATCH_SIZE ?? 128);
const MAX_CO_PPM = 200;
const SEQUENCE_STRIDE = Number(process.env.GRU_SEQUENCE_STRIDE ?? 10);
const SEED = Number(process.env.CO_COMPARE_SEED ?? JSON.parse(readFileSync(EXISTING_SUMMARY_PATH, 'utf8')).selectedSeed ?? 19);

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
    const coPpm = Number.isFinite(headerAlignedCo) ? headerAlignedCo : Number(cols.at(-1));
    const rawTimestamp = timeIndex >= 0 ? Number(cols[timeIndex]) : rowIndex * 2;
    return {
      timestampSeconds: timestampIsMs ? rawTimestamp / 1000 : rawTimestamp,
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

const readSampleGroups = () => (
  readdirSync(DATA_DIR)
    .filter((fileName) => fileName.toLowerCase().endsWith('.csv'))
    .sort()
    .map((fileName) => [fileName, parseCsv(resolve(DATA_DIR, fileName))])
);

const buildSequences = (sampleGroups) => {
  const sequences = [];
  for (const [sourceFile, samples] of sampleGroups) {
    if (samples.length <= TIME_STEPS + FORECAST_HORIZON_STEPS) {
      continue;
    }
    for (
      let start = 0;
      start + TIME_STEPS + FORECAST_HORIZON_STEPS < samples.length;
      start += SEQUENCE_STRIDE
    ) {
      const target = samples[start + TIME_STEPS + FORECAST_HORIZON_STEPS];
      sequences.push({
        window: samples.slice(start, start + TIME_STEPS).map((sample) => sample.features),
        target: target.coPpm,
        sourceFile,
      });
    }
  }
  return sequences;
};

const unzipSequences = (sequences) => {
  const windows = [];
  const targets = [];
  const sourceFiles = [];
  for (const sequence of sequences) {
    windows.push(sequence.window);
    targets.push(sequence.target);
    sourceFiles.push(sequence.sourceFile);
  }
  return { windows, targets, sourceFiles };
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

const computeVectorStats = (values) => {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
  return { mean, std: Math.sqrt(variance) || 1 };
};

const normalizeWindows = (windows, stats) => windows.map((window) => (
  window.map((row) => row.map((value, index) => (value - stats.mean[index]) / stats.std[index]))
));

const normalizeTargets = (targets, stats) => targets.map((value) => (value - stats.mean) / stats.std);
const denormalizeTargets = (targets, stats) => targets.map((value) => (value * stats.std) + stats.mean);
const lastWindowCoValues = (windows) => windows.map((window) => window[TIME_STEPS - 1][0]);

const calculateMetrics = (predictions, actuals) => {
  const mae = predictions.reduce((sum, prediction, index) => (
    sum + Math.abs(prediction - actuals[index])
  ), 0) / predictions.length;
  const rmse = Math.sqrt(predictions.reduce((sum, prediction, index) => (
    sum + ((actuals[index] - prediction) ** 2)
  ), 0) / predictions.length);
  const actualMean = actuals.reduce((sum, value) => sum + value, 0) / actuals.length;
  const ssRes = predictions.reduce((sum, prediction, index) => (
    sum + ((actuals[index] - prediction) ** 2)
  ), 0);
  const ssTot = actuals.reduce((sum, actual) => sum + ((actual - actualMean) ** 2), 0);
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
    sum + ((prediction - predictedMean) ** 2)
  ), 0);
  if (variance === 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, covariance / variance));
};

const buildModel = (architecture) => {
  const model = tf.sequential();
  if (architecture === 'Dense Baseline') {
    model.add(tf.layers.flatten({ inputShape: [TIME_STEPS, 5] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  } else if (architecture === '1D CNN') {
    model.add(tf.layers.conv1d({ inputShape: [TIME_STEPS, 5], filters: 24, kernelSize: 5, activation: 'relu', padding: 'same' }));
    model.add(tf.layers.globalAveragePooling1d());
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  } else if (architecture === 'Temporal CNN (TCN)') {
    model.add(tf.layers.conv1d({ inputShape: [TIME_STEPS, 5], filters: 24, kernelSize: 3, activation: 'relu', padding: 'same' }));
    model.add(tf.layers.conv1d({ filters: 24, kernelSize: 3, activation: 'relu', padding: 'same' }));
    model.add(tf.layers.globalAveragePooling1d());
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  } else if (architecture === 'LSTM') {
    model.add(tf.layers.lstm({ inputShape: [TIME_STEPS, 5], units: 48, returnSequences: false, recurrentInitializer: 'glorotUniform' }));
    model.add(tf.layers.dropout({ rate: 0.15 }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
  } else if (architecture === 'GRU') {
    model.add(tf.layers.gru({ inputShape: [TIME_STEPS, 5], units: 48, returnSequences: false, recurrentInitializer: 'glorotUniform' }));
    model.add(tf.layers.dropout({ rate: 0.15 }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
  } else {
    throw new Error(`Unsupported architecture: ${architecture}`);
  }
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

  const sequences = buildSequences(readSampleGroups());
  const { windows, targets } = unzipSequences(sequences);
  const testCount = Math.max(1, Math.round(windows.length * TEST_SPLIT));
  const trainCount = windows.length - testCount;
  const indices = shuffledIndices(windows.length, SEED);
  const testIndices = indices.slice(0, testCount).sort((a, b) => a - b);
  const testIndexSet = new Set(testIndices);
  const trainIndices = indices.filter((index) => !testIndexSet.has(index));

  const trainWindows = takeByIndex(windows, trainIndices);
  const testWindows = takeByIndex(windows, testIndices);
  const trainTargets = takeByIndex(targets, trainIndices);
  const testTargets = takeByIndex(targets, testIndices);
  const trainAnchors = lastWindowCoValues(trainWindows);
  const testAnchors = lastWindowCoValues(testWindows);
  const trainResidualTargets = trainTargets.map((target, index) => target - trainAnchors[index]);
  const testResidualTargets = testTargets.map((target, index) => target - testAnchors[index]);
  const persistenceMetrics = calculateMetrics(testAnchors, testTargets);
  const featureStats = computeStats(trainWindows);
  const targetStats = computeVectorStats(trainResidualTargets);
  const normalizedTrain = normalizeWindows(trainWindows, featureStats);
  const normalizedTest = normalizeWindows(testWindows, featureStats);
  const normalizedTrainTargets = normalizeTargets(trainResidualTargets, targetStats);

  const xsTrain = tf.tensor3d(normalizedTrain, [trainCount, TIME_STEPS, 5], 'float32');
  const ysTrain = tf.tensor2d(normalizedTrainTargets, [trainCount, 1], 'float32');
  const xsTest = tf.tensor3d(normalizedTest, [testCount, TIME_STEPS, 5], 'float32');
  const results = [{
    model: 'Persistence Baseline',
    testMae: persistenceMetrics.mae,
    testRmse: persistenceMetrics.rmse,
    testR2: persistenceMetrics.r2,
  }];

  for (const architecture of ['Dense Baseline', '1D CNN', 'Temporal CNN (TCN)', 'LSTM', 'GRU']) {
    const model = buildModel(architecture);
    const history = await model.fit(xsTrain, ysTrain, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationSplit: 0.15,
      shuffle: true,
      verbose: 0,
    });
    const normalizedTrainPredictions = Array.from(model.predict(xsTrain).dataSync());
    const trainResidualPredictions = denormalizeTargets(normalizedTrainPredictions, targetStats);
    const residualWeight = calibrateResidualWeight(trainResidualPredictions, trainResidualTargets);
    const normalizedPredictions = Array.from(model.predict(xsTest).dataSync());
    const residualPredictions = denormalizeTargets(normalizedPredictions, targetStats);
    const predictions = residualPredictions.map((residual, index) => testAnchors[index] + (residualWeight * residual));
    const metrics = calculateMetrics(predictions, testTargets);
    results.push({
      model: architecture,
      testMae: metrics.mae,
      testRmse: metrics.rmse,
      testR2: metrics.r2,
      calibratedResidualWeight: residualWeight,
      finalTrainMae: history.history.mae.at(-1) * targetStats.std,
      finalValidationMae: history.history.val_mae.at(-1) * targetStats.std,
    });
    model.dispose();
    console.log(JSON.stringify(results.at(-1)));
  }

  xsTrain.dispose();
  ysTrain.dispose();
  xsTest.dispose();

  const output = {
    generatedAt: new Date().toISOString(),
    seed: SEED,
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    testSplit: TEST_SPLIT,
    forecastHorizonSteps: FORECAST_HORIZON_STEPS,
    sequenceCount: windows.length,
    trainSequenceCount: trainCount,
    testSequenceCount: testCount,
    metric: 'testR2',
    note: 'CO forecasting is a regression task, so test R2 is used as the accuracy proxy.',
    modelComparison: results.sort((a, b) => b.testR2 - a.testR2),
  };
  writeFileSync(resolve(OUTPUT_DIR, 'co_forecasting_model_comparison.json'), JSON.stringify(output, null, 2));
  console.log(JSON.stringify(output, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
