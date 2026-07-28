import { createStore } from 'zustand/vanilla';
import type { PredictionResult, TelemetrySample } from './telemetryTypes';

type TelemetryState = {
  latestSample: TelemetrySample | null;
  latestPrediction: PredictionResult | null;
  chartSamples: TelemetrySample[];
  connectionState: 'disconnected' | 'connecting' | 'connected';
};

type TelemetryActions = {
  setConnectionState: (connectionState: TelemetryState['connectionState']) => void;
  publishUiSample: (sample: TelemetrySample) => void;
  publishPrediction: (prediction: PredictionResult) => void;
  reset: () => void;
};

export type TelemetryStore = TelemetryState & TelemetryActions;

const MAX_CHART_SAMPLES = 10 * 60;

export const telemetryStore = createStore<TelemetryStore>((set) => ({
  latestSample: null,
  latestPrediction: null,
  chartSamples: [],
  connectionState: 'disconnected',

  setConnectionState: (connectionState) => set({ connectionState }),

  publishUiSample: (sample) => set((state) => {
    const chartSamples = [...state.chartSamples, sample];
    if (chartSamples.length > MAX_CHART_SAMPLES) {
      chartSamples.splice(0, chartSamples.length - MAX_CHART_SAMPLES);
    }
    return {
      latestSample: sample,
      chartSamples,
    };
  }),

  publishPrediction: (prediction) => set({ latestPrediction: prediction }),

  reset: () => set({
    latestSample: null,
    latestPrediction: null,
    chartSamples: [],
    connectionState: 'disconnected',
  }),
}));

export const createThrottledUiPublisher = (minIntervalMs = 250) => {
  let lastPublishAt = 0;

  return (sample: TelemetrySample): void => {
    const now = performance.now();
    if (now - lastPublishAt < minIntervalMs) {
      return;
    }

    lastPublishAt = now;
    telemetryStore.getState().publishUiSample(sample);
  };
};
