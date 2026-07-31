export type TelemetrySample = {
  timestampMs: number;
  coPpm: number;
  bpm: number;
  accelX: number;
  accelY: number;
  accelZ: number;
};

export type PredictionResult = {
  predictedCoPpm: number;
  horizonSeconds: number;
  isCriticalCo: boolean;
  isFallDetected: boolean;
  fallSvmDecisionScore: number;
  fallSvmPredictedClass: number;
  tachycardiaRiskScore: number;
  isAcutePhysiologicalDistress: boolean;
  accelerationMagnitudeG: number;
  timestampMs: number;
};

export type TelemetryListener = (sample: TelemetrySample) => void;
export type PredictionListener = (
  prediction: PredictionResult,
  sample: TelemetrySample,
) => void;

export const TELEMETRY_FEATURE_COUNT = 5;

export const telemetryToFeatureVector = (sample: TelemetrySample): number[] => [
  sample.coPpm,
  sample.bpm,
  sample.accelX,
  sample.accelY,
  sample.accelZ,
];
