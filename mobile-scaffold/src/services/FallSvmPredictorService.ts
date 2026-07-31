import fallSvmArtifact from '../models/fall-svm.json';
import type { TelemetrySample } from './telemetryTypes';

type FallSvmArtifact = {
  modelType: 'rbf_svc';
  classes: number[];
  gamma: number;
  intercept: number[];
  dualCoef: number[][];
  supportVectors: number[][];
  scalerMean: number[];
  scalerScale: number[];
  featureNames: string[];
};

export type FallSvmPrediction = {
  isFallDetected: boolean;
  predictedClass: number;
  decisionScore: number;
};

const artifact = fallSvmArtifact as FallSvmArtifact;
const SIGNALS = ['magnitude', 'x', 'y', 'z'] as const;

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

const std = (values: number[]) => {
  const valueMean = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - valueMean) ** 2)));
};

const percentile = (values: number[], percentileValue: number) => {
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (percentileValue / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) {
    return sorted[lower];
  }
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
};

const skew = (values: number[]) => {
  const n = values.length;
  if (n <= 2) {
    return 0;
  }
  const valueMean = mean(values);
  const m2 = mean(values.map((value) => (value - valueMean) ** 2));
  if (m2 === 0) {
    return 0;
  }
  const m3 = mean(values.map((value) => (value - valueMean) ** 3));
  return (Math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / (m2 ** 1.5));
};

const kurtosis = (values: number[]) => {
  const n = values.length;
  if (n <= 3) {
    return 0;
  }
  const valueMean = mean(values);
  const m2 = mean(values.map((value) => (value - valueMean) ** 2));
  if (m2 === 0) {
    return 0;
  }
  const m4 = mean(values.map((value) => (value - valueMean) ** 4));
  const biasedExcess = m4 / (m2 ** 2) - 3;
  return ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * biasedExcess + 6);
};

const correlation = (first: number[], second: number[]) => {
  const firstMean = mean(first);
  const secondMean = mean(second);
  let numerator = 0;
  let firstSum = 0;
  let secondSum = 0;

  for (let i = 0; i < first.length; i += 1) {
    const firstDelta = first[i] - firstMean;
    const secondDelta = second[i] - secondMean;
    numerator += firstDelta * secondDelta;
    firstSum += firstDelta * firstDelta;
    secondSum += secondDelta * secondDelta;
  }

  const denominator = Math.sqrt(firstSum * secondSum);
  return denominator === 0 ? 0 : numerator / denominator;
};

const momentFeatures = (values: number[]) => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return [
    mean(values),
    std(values),
    min,
    max,
    percentile(values, 50),
    percentile(values, 25),
    percentile(values, 75),
    max - min,
    Math.sqrt(mean(values.map((value) => value * value))),
    skew(values),
    kurtosis(values),
  ];
};

export class FallSvmPredictorService {
  predict(samples: TelemetrySample[]): FallSvmPrediction {
    if (samples.length < 5) {
      return {
        isFallDetected: false,
        predictedClass: 0,
        decisionScore: Number.NEGATIVE_INFINITY,
      };
    }

    const features = this.extractFeatures(samples);
    const scaled = features.map((value, index) => (
      (value - artifact.scalerMean[index]) / artifact.scalerScale[index]
    ));
    const decisionScore = this.decisionFunction(scaled);
    const predictedClass = decisionScore > 0 ? artifact.classes[1] : artifact.classes[0];

    return {
      isFallDetected: predictedClass === 1,
      predictedClass,
      decisionScore,
    };
  }

  getModelSummary() {
    return {
      modelType: artifact.modelType,
      classes: artifact.classes,
      supportVectorCount: artifact.supportVectors.length,
      featureCount: artifact.featureNames.length,
      gamma: artifact.gamma,
      featureNames: artifact.featureNames,
    };
  }

  private extractFeatures(samples: TelemetrySample[]): number[] {
    const x = samples.map((sample) => sample.accelX).filter(Number.isFinite);
    const y = samples.map((sample) => sample.accelY).filter(Number.isFinite);
    const z = samples.map((sample) => sample.accelZ).filter(Number.isFinite);
    const rows = Math.min(x.length, y.length, z.length);
    const axes = {
      x: x.slice(0, rows),
      y: y.slice(0, rows),
      z: z.slice(0, rows),
      magnitude: Array.from({ length: rows }, (_, index) => Math.hypot(x[index], y[index], z[index])),
    };

    const values: number[] = [];
    for (const signal of SIGNALS) {
      values.push(...momentFeatures(axes[signal]));
    }

    const magnitude = axes.magnitude;
    const jerk = Array.from({ length: Math.max(0, magnitude.length - 1) }, (_, index) => (
      magnitude[index + 1] - magnitude[index]
    ));

    values.push(
      mean(jerk.map(Math.abs)),
      std(jerk),
      Math.max(...jerk.map(Math.abs)),
      correlation(axes.x, axes.y),
      correlation(axes.x, axes.z),
      correlation(axes.y, axes.z),
    );

    return values.map((value) => (Number.isFinite(value) ? value : 0));
  }

  private decisionFunction(scaledFeatures: number[]): number {
    const coefficients = artifact.dualCoef[0];
    let score = artifact.intercept[0];

    for (let i = 0; i < artifact.supportVectors.length; i += 1) {
      const supportVector = artifact.supportVectors[i];
      let squaredDistance = 0;
      for (let j = 0; j < scaledFeatures.length; j += 1) {
        const delta = scaledFeatures[j] - supportVector[j];
        squaredDistance += delta * delta;
      }
      score += coefficients[i] * Math.exp(-artifact.gamma * squaredDistance);
    }

    return score;
  }
}

export const fallSvmPredictor = new FallSvmPredictorService();
