import type { TelemetrySample } from './telemetryTypes';

export type TachycardiaRiskConfig = {
  alpha: number;
  beta: number;
  restingBpm: number;
  tachycardiaBpm: number;
  acuteRiseBpmPerMinute: number;
  distressThreshold: number;
};

export type TachycardiaRiskResult = {
  score: number;
  alpha: number;
  beta: number;
  heartRateElevationComponent: number;
  heartRateRiseComponent: number;
  heartRateRiseBpmPerMinute: number;
  distressThreshold: number;
  isAcutePhysiologicalDistress: boolean;
};

export const DEFAULT_TACHYCARDIA_RISK_CONFIG: TachycardiaRiskConfig = {
  alpha: 0.6,
  beta: 0.4,
  restingBpm: 80,
  tachycardiaBpm: 100,
  acuteRiseBpmPerMinute: 20,
  distressThreshold: 0.7,
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

export const computeTachycardiaRisk = (
  samples: TelemetrySample[],
  config: TachycardiaRiskConfig = DEFAULT_TACHYCARDIA_RISK_CONFIG,
): TachycardiaRiskResult => {
  const latest = samples[samples.length - 1];
  if (!latest) {
    return {
      score: 0,
      alpha: config.alpha,
      beta: config.beta,
      heartRateElevationComponent: 0,
      heartRateRiseComponent: 0,
      heartRateRiseBpmPerMinute: 0,
      distressThreshold: config.distressThreshold,
      isAcutePhysiologicalDistress: false,
    };
  }

  const heartRateElevationComponent = clamp01(
    (latest.bpm - config.restingBpm) / (config.tachycardiaBpm - config.restingBpm),
  );

  const earliest = samples[0];
  const elapsedMinutes = earliest && latest.timestampMs > earliest.timestampMs
    ? (latest.timestampMs - earliest.timestampMs) / 60_000
    : 0;
  const heartRateRiseBpmPerMinute = elapsedMinutes > 0
    ? (latest.bpm - earliest.bpm) / elapsedMinutes
    : 0;
  const heartRateRiseComponent = clamp01(
    heartRateRiseBpmPerMinute / config.acuteRiseBpmPerMinute,
  );

  const score = clamp01(
    config.alpha * heartRateElevationComponent
    + config.beta * heartRateRiseComponent,
  );

  return {
    score,
    alpha: config.alpha,
    beta: config.beta,
    heartRateElevationComponent,
    heartRateRiseComponent,
    heartRateRiseBpmPerMinute,
    distressThreshold: config.distressThreshold,
    isAcutePhysiologicalDistress: score >= config.distressThreshold,
  };
};
