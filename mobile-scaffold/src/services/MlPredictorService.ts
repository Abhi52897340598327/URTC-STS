import * as tf from '@tensorflow/tfjs';
import {
  measurePredictLatency,
  type PredictLatencySummary,
} from './EvaluationInstrumentationService';
import {
  TELEMETRY_FEATURE_COUNT,
  type PredictionListener,
  type PredictionResult,
  type TelemetrySample,
  telemetryToFeatureVector,
} from './telemetryTypes';
import {
  DEFAULT_TACHYCARDIA_RISK_CONFIG,
  computeTachycardiaRisk,
  type TachycardiaRiskConfig,
} from './TachycardiaRiskService';
import { fallSvmPredictor } from './FallSvmPredictorService';

type MlPredictorOptions = {
  timeSteps?: number;
  forecastHorizonSteps?: number;
  criticalCoPpm?: number;
  tachycardiaRiskConfig?: TachycardiaRiskConfig;
  modelUrl?: string;
};

type TrainingOptions = {
  epochs?: number;
  batchSize?: number;
  validationSplit?: number;
  shuffle?: boolean;
  learningRate?: number;
};

type FeatureStats = {
  mean: number[];
  std: number[];
};

export type MlArchitectureSummary = {
  framework: 'TensorFlow.js';
  modelType: 'GRU';
  storageUrl: string;
  inputShape: [number, number];
  inputWindowLengthTimesteps: number;
  inputWindowLengthSecondsAt10Hz: number;
  forecastHorizonSteps: number;
  forecastHorizonSecondsAt10Hz: number;
  recurrentLayers: number;
  gruUnits: number;
  denseUnits: number[];
  dropoutRate: number;
  loss: string;
  optimizer: string;
  defaultLearningRate: number;
  defaultEpochs: number;
  defaultBatchSize: number;
  defaultValidationSplit: number;
  criticalCoPpm: number;
  fallDetectionModelType: string;
  fallSvmSupportVectorCount: number;
  fallSvmFeatureCount: number;
  fallSvmGamma: number;
  tachycardiaAlpha: number;
  tachycardiaBeta: number;
  tachycardiaDistressThreshold: number;
  tachycardiaRestingBpm: number;
  tachycardiaBpm: number;
  tachycardiaAcuteRiseBpmPerMinute: number;
};

const DEFAULT_MODEL_URL = 'indexeddb://co-gru-model';

/**
 * Builds, trains, persists, and runs a GRU forecaster on live CODetect telemetry.
 *
 * The model is trained from scratch. No pretrained weights are assumed.
 * Input tensor shape: [samples, timeSteps, 5]
 * Target tensor shape: [samples, 1], predicting future CO PPM.
 */
export class MlPredictorService {
  private readonly timeSteps: number;
  private readonly forecastHorizonSteps: number;
  private readonly criticalCoPpm: number;
  private readonly tachycardiaRiskConfig: TachycardiaRiskConfig;
  private readonly modelUrl: string;

  private model: tf.LayersModel | null = null;
  private featureStats: FeatureStats | null = null;
  private readonly liveWindow: TelemetrySample[] = [];
  private readonly listeners = new Set<PredictionListener>();
  private inferenceBusy = false;

  constructor(options: MlPredictorOptions = {}) {
    this.timeSteps = options.timeSteps ?? 30;
    this.forecastHorizonSteps = options.forecastHorizonSteps ?? 30;
    this.criticalCoPpm = options.criticalCoPpm ?? 70;
    this.tachycardiaRiskConfig = options.tachycardiaRiskConfig ?? DEFAULT_TACHYCARDIA_RISK_CONFIG;
    this.modelUrl = options.modelUrl ?? DEFAULT_MODEL_URL;
  }

  async init(): Promise<void> {
    await tf.ready();
    await this.tryLoadModel();
  }

  subscribe(listener: PredictionListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  hasModel(): boolean {
    return this.model !== null;
  }

  getArchitectureSummary(): MlArchitectureSummary {
    const fallSvmSummary = fallSvmPredictor.getModelSummary();
    return {
      framework: 'TensorFlow.js',
      modelType: 'GRU',
      storageUrl: this.modelUrl,
      inputShape: [this.timeSteps, TELEMETRY_FEATURE_COUNT],
      inputWindowLengthTimesteps: this.timeSteps,
      inputWindowLengthSecondsAt10Hz: this.timeSteps / 10,
      forecastHorizonSteps: this.forecastHorizonSteps,
      forecastHorizonSecondsAt10Hz: this.forecastHorizonSteps / 10,
      recurrentLayers: 1,
      gruUnits: 48,
      denseUnits: [24, 1],
      dropoutRate: 0.15,
      loss: 'meanSquaredError',
      optimizer: 'Adam',
      defaultLearningRate: 0.001,
      defaultEpochs: 30,
      defaultBatchSize: 32,
      defaultValidationSplit: 0.15,
      criticalCoPpm: this.criticalCoPpm,
      fallDetectionModelType: 'RBF SVM',
      fallSvmSupportVectorCount: fallSvmSummary.supportVectorCount,
      fallSvmFeatureCount: fallSvmSummary.featureCount,
      fallSvmGamma: fallSvmSummary.gamma,
      tachycardiaAlpha: this.tachycardiaRiskConfig.alpha,
      tachycardiaBeta: this.tachycardiaRiskConfig.beta,
      tachycardiaDistressThreshold: this.tachycardiaRiskConfig.distressThreshold,
      tachycardiaRestingBpm: this.tachycardiaRiskConfig.restingBpm,
      tachycardiaBpm: this.tachycardiaRiskConfig.tachycardiaBpm,
      tachycardiaAcuteRiseBpmPerMinute: this.tachycardiaRiskConfig.acuteRiseBpmPerMinute,
    };
  }

  async measureSinglePredictLatency(windowSamples: TelemetrySample[], runs = 100): Promise<PredictLatencySummary> {
    if (!this.model || !this.featureStats) {
      throw new Error('Model and feature stats must be initialized before benchmarking predict latency');
    }
    if (windowSamples.length !== this.timeSteps) {
      throw new Error(`Expected ${this.timeSteps} samples, got ${windowSamples.length}`);
    }

    const input = tf.tensor3d(
      this.flattenNormalizedWindow(windowSamples),
      [1, this.timeSteps, TELEMETRY_FEATURE_COUNT],
      'float32',
    );

    try {
      return await measurePredictLatency(this.model, input, runs);
    } finally {
      input.dispose();
    }
  }

  buildModel(learningRate = 0.001): tf.LayersModel {
    const model = tf.sequential();

    model.add(tf.layers.gru({
      inputShape: [this.timeSteps, TELEMETRY_FEATURE_COUNT],
      units: 48,
      returnSequences: false,
      recurrentInitializer: 'glorotUniform',
    }));
    model.add(tf.layers.dropout({ rate: 0.15 }));
    model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    return model;
  }

  async trainModelFromData(
    historicalData: TelemetrySample[],
    options: TrainingOptions = {},
  ): Promise<tf.History> {
    if (historicalData.length < this.timeSteps + this.forecastHorizonSteps + 1) {
      throw new Error(
        `Need at least ${this.timeSteps + this.forecastHorizonSteps + 1} samples, got ${historicalData.length}`,
      );
    }

    const cleaned = historicalData.filter((sample) => Number.isFinite(sample.coPpm));
    this.featureStats = this.computeFeatureStats(cleaned);

    const { xsData, ysData, sampleCount } = this.createTrainingArrays(cleaned);
    this.model?.dispose();
    this.model = this.buildModel(options.learningRate);

    const xs = tf.tensor3d(xsData, [sampleCount, this.timeSteps, TELEMETRY_FEATURE_COUNT], 'float32');
    const ys = tf.tensor2d(ysData, [sampleCount, 1], 'float32');

    try {
      const history = await this.model.fit(xs, ys, {
        epochs: options.epochs ?? 30,
        batchSize: options.batchSize ?? 32,
        validationSplit: options.validationSplit ?? 0.15,
        shuffle: options.shuffle ?? true,
        callbacks: {
          onEpochEnd: async () => {
            // Yield between epochs so the browser event loop and Capacitor bridge
            // are not starved during on-device training.
            await tf.nextFrame();
          },
        },
      });

      await this.model.save(this.modelUrl);
      this.persistFeatureStats(this.featureStats);
      return history;
    } finally {
      xs.dispose();
      ys.dispose();
    }
  }

  /**
   * Push one live 10 Hz sample through the GRU once enough history exists.
   *
   * All tensors used for inference are created inside tf.tidy() and converted
   * to a plain number before returning. No Tensor escapes the hot path.
   */
  async ingestLiveSample(sample: TelemetrySample): Promise<PredictionResult | null> {
    this.liveWindow.push(sample);
    if (this.liveWindow.length > this.timeSteps) {
      this.liveWindow.splice(0, this.liveWindow.length - this.timeSteps);
    }

    if (!this.model || !this.featureStats || this.liveWindow.length < this.timeSteps || this.inferenceBusy) {
      return null;
    }

    this.inferenceBusy = true;
    try {
      const windowSnapshot = this.liveWindow.slice();
      const predictedCoPpm = tf.tidy(() => {
        const normalized = this.flattenNormalizedWindow(windowSnapshot);
        const input = tf.tensor3d(
          normalized,
          [1, this.timeSteps, TELEMETRY_FEATURE_COUNT],
          'float32',
        );
        const output = this.model!.predict(input) as tf.Tensor;
        return output.dataSync()[0];
      });

      const accelerationMagnitudeG = this.accelerationMagnitude(sample);
      const fallSvmPrediction = fallSvmPredictor.predict(windowSnapshot);
      const tachycardiaRisk = computeTachycardiaRisk(windowSnapshot, this.tachycardiaRiskConfig);
      const prediction: PredictionResult = {
        predictedCoPpm,
        horizonSeconds: this.forecastHorizonSteps / 10,
        isCriticalCo: predictedCoPpm >= this.criticalCoPpm,
        isFallDetected: fallSvmPrediction.isFallDetected,
        fallSvmDecisionScore: fallSvmPrediction.decisionScore,
        fallSvmPredictedClass: fallSvmPrediction.predictedClass,
        tachycardiaRiskScore: tachycardiaRisk.score,
        isAcutePhysiologicalDistress: tachycardiaRisk.isAcutePhysiologicalDistress,
        accelerationMagnitudeG,
        timestampMs: Date.now(),
      };

      for (const listener of this.listeners) {
        listener(prediction, sample);
      }

      return prediction;
    } finally {
      this.inferenceBusy = false;
    }
  }

  async dispose(): Promise<void> {
    this.model?.dispose();
    this.model = null;
    this.liveWindow.length = 0;
    this.listeners.clear();
    await tf.nextFrame();
  }

  loadCoCsvForTraining(csvText: string): TelemetrySample[] {
    const lines = csvText.trim().split(/\r?\n/);
    const header = lines.shift()?.split(',').map((name) => name.trim().toLowerCase()) ?? [];
    const coIndex = header.findIndex((name) => name === 'co_ppm' || name === 'co(ppm)' || name === 'co ppm');
    const timeIndex = header.findIndex((name) => name.startsWith('timestamp'));

    if (coIndex < 0) {
      throw new Error('CSV does not contain a CO column');
    }

    return lines
      .map((line, index) => {
        const cols = line.split(',');
        const coPpm = Number(cols[coIndex]);
        return {
          timestampMs: timeIndex >= 0 ? Number(cols[timeIndex]) : index * 100,
          coPpm,
          // Historical CO CSVs in data/co-data do not include wearable signals.
          // Keep the feature shape identical to live data and retrain once full
          // synchronized CODetect logs are available.
          bpm: 0,
          accelX: 0,
          accelY: 0,
          accelZ: 0,
        };
      })
      .filter((sample) => Number.isFinite(sample.coPpm));
  }

  private async tryLoadModel(): Promise<void> {
    try {
      this.model = await tf.loadLayersModel(this.modelUrl);
      const persistedStats = localStorage.getItem('co-gru-feature-stats');
      if (persistedStats) {
        this.featureStats = JSON.parse(persistedStats) as FeatureStats;
      }
    } catch {
      this.model = null;
      this.featureStats = null;
    }
  }

  private createTrainingArrays(data: TelemetrySample[]): {
    xsData: number[];
    ysData: number[];
    sampleCount: number;
  } {
    const xsData: number[] = [];
    const ysData: number[] = [];

    for (let start = 0; start + this.timeSteps + this.forecastHorizonSteps < data.length; start += 1) {
      const inputWindow = data.slice(start, start + this.timeSteps);
      xsData.push(...this.flattenNormalizedWindow(inputWindow));

      const targetSample = data[start + this.timeSteps + this.forecastHorizonSteps];
      ysData.push(targetSample.coPpm);
    }

    return {
      xsData,
      ysData,
      sampleCount: ysData.length,
    };
  }

  private flattenNormalizedWindow(window: TelemetrySample[]): number[] {
    if (!this.featureStats) {
      throw new Error('Feature stats are not initialized');
    }

    const values: number[] = [];
    for (const sample of window) {
      const vector = telemetryToFeatureVector(sample);
      for (let i = 0; i < TELEMETRY_FEATURE_COUNT; i += 1) {
        values.push((vector[i] - this.featureStats.mean[i]) / this.featureStats.std[i]);
      }
    }
    return values;
  }

  private computeFeatureStats(data: TelemetrySample[]): FeatureStats {
    const mean = new Array(TELEMETRY_FEATURE_COUNT).fill(0);
    const std = new Array(TELEMETRY_FEATURE_COUNT).fill(0);

    for (const sample of data) {
      const vector = telemetryToFeatureVector(sample);
      for (let i = 0; i < TELEMETRY_FEATURE_COUNT; i += 1) {
        mean[i] += vector[i];
      }
    }

    for (let i = 0; i < TELEMETRY_FEATURE_COUNT; i += 1) {
      mean[i] /= data.length;
    }

    for (const sample of data) {
      const vector = telemetryToFeatureVector(sample);
      for (let i = 0; i < TELEMETRY_FEATURE_COUNT; i += 1) {
        const delta = vector[i] - mean[i];
        std[i] += delta * delta;
      }
    }

    for (let i = 0; i < TELEMETRY_FEATURE_COUNT; i += 1) {
      std[i] = Math.sqrt(std[i] / data.length) || 1;
    }

    return { mean, std };
  }

  private persistFeatureStats(stats: FeatureStats): void {
    localStorage.setItem('co-gru-feature-stats', JSON.stringify(stats));
  }

  private accelerationMagnitude(sample: TelemetrySample): number {
    return Math.sqrt(
      sample.accelX * sample.accelX
      + sample.accelY * sample.accelY
      + sample.accelZ * sample.accelZ,
    );
  }
}
