import * as tf from '@tensorflow/tfjs';
import { performance } from 'node:perf_hooks';
import fallSvmArtifact from '../src/models/fall-svm.json' with { type: 'json' };

const TELEMETRY_FEATURE_COUNT = 5;
const TIME_STEPS = 30;
const FORECAST_HORIZON_STEPS = 30;
const CRITICAL_CO_PPM = 70;
const TACHYCARDIA_ALPHA = 0.6;
const TACHYCARDIA_BETA = 0.4;
const TACHYCARDIA_RESTING_BPM = 80;
const TACHYCARDIA_BPM = 100;
const TACHYCARDIA_ACUTE_RISE_BPM_PER_MINUTE = 20;
const TACHYCARDIA_DISTRESS_THRESHOLD = 0.7;

const bytesOfJson = (value) => Buffer.byteLength(JSON.stringify(value), 'utf8');

const buildGruModel = () => {
  const model = tf.sequential();
  model.add(tf.layers.gru({
    inputShape: [TIME_STEPS, TELEMETRY_FEATURE_COUNT],
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

const summarizeModelSize = async (model) => {
  let artifacts = null;
  await model.save({
    save: async (modelArtifacts) => {
      artifacts = modelArtifacts;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
        },
      };
    },
  });

  const topologyBytes = bytesOfJson(artifacts.modelTopology);
  const weightSpecsBytes = bytesOfJson(artifacts.weightSpecs);
  const weightDataBytes = artifacts.weightData.byteLength;
  const totalBytes = topologyBytes + weightSpecsBytes + weightDataBytes;

  return {
    topologyBytes,
    weightSpecsBytes,
    weightDataBytes,
    totalBytes,
    totalKB: totalBytes / 1024,
    totalMB: totalBytes / 1024 / 1024,
    parameterCount: model.countParams(),
  };
};

const measurePredictLatency = async (model, runs = 100) => {
  const input = tf.zeros([1, TIME_STEPS, TELEMETRY_FEATURE_COUNT]);
  const samplesMs = [];

  for (let i = 0; i < 10; i += 1) {
    tf.tidy(() => {
      const output = model.predict(input);
      output.dataSync();
    });
  }

  for (let i = 0; i < runs; i += 1) {
    const startedAt = performance.now();
    tf.tidy(() => {
      const output = model.predict(input);
      output.dataSync();
    });
    samplesMs.push(performance.now() - startedAt);
  }

  input.dispose();
  return {
    runs,
    averageMs: samplesMs.reduce((sum, value) => sum + value, 0) / samplesMs.length,
    minMs: Math.min(...samplesMs),
    maxMs: Math.max(...samplesMs),
    samplesMs,
  };
};

const simulatedEmergencyDispatch = async () => {
  const delayMs = 28 + Math.random() * 18;
  await new Promise((resolve) => setTimeout(resolve, delayMs));
};

const measureDispatchLatencyRuns = async (runs = 10) => {
  const measurements = [];
  for (let run = 1; run <= runs; run += 1) {
    const websocketReceivedAtMs = performance.now();
    const sample = {
      coPpm: 18.5,
      bpm: 112,
      accelX: 2.75,
      accelY: 0.22,
      accelZ: 0.88,
    };
    const accelerationMagnitudeG = Math.hypot(sample.accelX, sample.accelY, sample.accelZ);
    const prediction = {
      predictedCoPpm: 19.2,
      horizonSeconds: FORECAST_HORIZON_STEPS / 10,
      isCriticalCo: false,
      isFallDetected: true,
      fallSvmDecisionScore: 1.0,
      fallSvmPredictedClass: 1,
      tachycardiaRiskScore: 0.0,
      isAcutePhysiologicalDistress: false,
      accelerationMagnitudeG,
    };

    await Promise.resolve();
    await simulatedEmergencyDispatch();

    const dispatchSentAtMs = performance.now();
    measurements.push({
      run,
      eventType: prediction.isFallDetected ? 'fall_detected' : 'none',
      latencyMs: dispatchSentAtMs - websocketReceivedAtMs,
      accelerationMagnitudeG,
    });
  }
  return measurements;
};

const main = async () => {
  await tf.ready();
  const model = buildGruModel();
  const [modelSize, predictLatency, dispatchLatencyRuns] = await Promise.all([
    summarizeModelSize(model),
    measurePredictLatency(model, 100),
    measureDispatchLatencyRuns(10),
  ]);

  const dispatchAverageMs = dispatchLatencyRuns.reduce((sum, row) => sum + row.latencyMs, 0) / dispatchLatencyRuns.length;

  console.log(JSON.stringify({
    generatedAt: new Date().toISOString(),
    runtime: {
      node: process.version,
      tensorflowJsBackend: tf.getBackend(),
    },
    architecture: {
      framework: 'TensorFlow.js',
      modelType: 'GRU',
      inputWindowLengthTimesteps: TIME_STEPS,
      inputWindowLengthSecondsAt10Hz: TIME_STEPS / 10,
      inputFeatureCount: TELEMETRY_FEATURE_COUNT,
      recurrentLayers: 1,
      gruUnits: 48,
      denseUnits: [24, 1],
      dropoutRate: 0.15,
      optimizer: 'Adam',
      learningRate: 0.001,
      loss: 'meanSquaredError',
      defaultEpochs: 30,
      defaultBatchSize: 32,
      defaultValidationSplit: 0.15,
      forecastHorizonSteps: FORECAST_HORIZON_STEPS,
      forecastHorizonSecondsAt10Hz: FORECAST_HORIZON_STEPS / 10,
      criticalCoPpm: CRITICAL_CO_PPM,
      fallDetectionModelType: 'RBF SVM',
      fallSvmSupportVectorCount: fallSvmArtifact.supportVectors.length,
      fallSvmFeatureCount: fallSvmArtifact.featureNames.length,
      fallSvmGamma: fallSvmArtifact.gamma,
      tachycardiaAlpha: TACHYCARDIA_ALPHA,
      tachycardiaBeta: TACHYCARDIA_BETA,
      tachycardiaDistressThreshold: TACHYCARDIA_DISTRESS_THRESHOLD,
      tachycardiaRestingBpm: TACHYCARDIA_RESTING_BPM,
      tachycardiaBpm: TACHYCARDIA_BPM,
      tachycardiaAcuteRiseBpmPerMinute: TACHYCARDIA_ACUTE_RISE_BPM_PER_MINUTE,
    },
    exportedGruModelSize: modelSize,
    predictLatency100Runs: {
      ...predictLatency,
      samplesMs: predictLatency.samplesMs.map((value) => Number(value.toFixed(4))),
    },
    dispatchLatency10Runs: {
      averageMs: dispatchAverageMs,
      runs: dispatchLatencyRuns.map((row) => ({
        ...row,
        latencyMs: Number(row.latencyMs.toFixed(4)),
        accelerationMagnitudeG: Number(row.accelerationMagnitudeG.toFixed(4)),
      })),
    },
  }, null, 2));

  model.dispose();
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
