import * as tf from '@tensorflow/tfjs';
import type { PredictionResult, TelemetrySample } from './telemetryTypes';

export type EmergencyDispatchPayload = {
  eventType: 'fall_detected' | 'critical_co' | 'acute_physiological_distress';
  timestampMs: number;
  predictedCoPpm: number;
  accelerationMagnitudeG: number;
  fallSvmDecisionScore: number;
  fallSvmPredictedClass: number;
  tachycardiaRiskScore: number;
  horizonSeconds: number;
};

export type DispatchTransport = (payload: EmergencyDispatchPayload) => Promise<void>;

export type DispatchLatencyMeasurement = {
  eventType: EmergencyDispatchPayload['eventType'];
  websocketReceivedAtMs: number;
  dispatchSentAtMs: number;
  latencyMs: number;
  payload: EmergencyDispatchPayload;
};

export type PredictLatencySummary = {
  runs: number;
  averageMs: number;
  minMs: number;
  maxMs: number;
  samplesMs: number[];
};

const nowMs = () => performance.now();

export const createEmergencyDispatchPayload = (
  sample: TelemetrySample,
  prediction: PredictionResult,
): EmergencyDispatchPayload => ({
  eventType: prediction.isFallDetected
    ? 'fall_detected'
    : prediction.isAcutePhysiologicalDistress
      ? 'acute_physiological_distress'
      : 'critical_co',
  timestampMs: Date.now(),
  predictedCoPpm: prediction.predictedCoPpm,
  accelerationMagnitudeG: prediction.accelerationMagnitudeG,
  fallSvmDecisionScore: prediction.fallSvmDecisionScore,
  fallSvmPredictedClass: prediction.fallSvmPredictedClass,
  tachycardiaRiskScore: prediction.tachycardiaRiskScore,
  horizonSeconds: prediction.horizonSeconds,
});

export const simulatedDispatchTransport: DispatchTransport = async (payload) => {
  const endpoint = localStorage.getItem('codetect-dispatch-endpoint');
  if (!endpoint) {
    await new Promise((resolve) => window.setTimeout(resolve, 35));
    return;
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Emergency dispatch failed with HTTP ${response.status}`);
  }
};

export const measureEmergencyDispatchLatency = async (
  websocketReceivedAtMs: number,
  sample: TelemetrySample,
  prediction: PredictionResult,
  transport: DispatchTransport = simulatedDispatchTransport,
): Promise<DispatchLatencyMeasurement> => {
  const payload = createEmergencyDispatchPayload(sample, prediction);
  await transport(payload);
  const dispatchSentAtMs = nowMs();

  return {
    eventType: payload.eventType,
    websocketReceivedAtMs,
    dispatchSentAtMs,
    latencyMs: dispatchSentAtMs - websocketReceivedAtMs,
    payload,
  };
};

export const measurePredictLatency = async (
  model: tf.LayersModel,
  input: tf.Tensor,
  runs = 100,
): Promise<PredictLatencySummary> => {
  const samplesMs: number[] = [];

  for (let i = 0; i < 5; i += 1) {
    tf.tidy(() => {
      const output = model.predict(input) as tf.Tensor;
      output.dataSync();
    });
    await tf.nextFrame();
  }

  for (let i = 0; i < runs; i += 1) {
    const startedAt = nowMs();
    tf.tidy(() => {
      const output = model.predict(input) as tf.Tensor;
      output.dataSync();
    });
    samplesMs.push(nowMs() - startedAt);
    await tf.nextFrame();
  }

  const total = samplesMs.reduce((sum, value) => sum + value, 0);
  return {
    runs,
    averageMs: total / runs,
    minMs: Math.min(...samplesMs),
    maxMs: Math.max(...samplesMs),
    samplesMs,
  };
};
