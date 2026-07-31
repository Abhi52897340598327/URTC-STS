import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles.css';
import type { PredictionResult, TelemetrySample } from './services/telemetryTypes';
import { computeTachycardiaRisk } from './services/TachycardiaRiskService';
import { fallSvmPredictor } from './services/FallSvmPredictorService';

type FuturePoint = {
  minute: number;
  coPpm: number;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const randomBetween = (min: number, max: number) => min + Math.random() * (max - min);

const createInitialSamples = (): TelemetrySample[] => {
  const now = Date.now();
  return Array.from({ length: 40 }, (_, index) => ({
    timestampMs: now - (40 - index) * 250,
    coPpm: randomBetween(8, 18),
    bpm: Math.round(randomBetween(68, 86)),
    accelX: randomBetween(-0.08, 0.08),
    accelY: randomBetween(-0.08, 0.08),
    accelZ: randomBetween(0.94, 1.06),
  }));
};

const simulateNextSample = (previous: TelemetrySample | null): TelemetrySample => {
  const baseCo = previous?.coPpm ?? 12;
  const drift = Math.sin(Date.now() / 18_000) * 0.45;
  const occasionalRise = Math.random() > 0.94 ? randomBetween(2.5, 6.5) : 0;
  const coPpm = clamp(baseCo + drift + randomBetween(-1.2, 1.5) + occasionalRise, 2, 95);

  return {
    timestampMs: Date.now(),
    coPpm,
    bpm: Math.round(clamp((previous?.bpm ?? 76) + randomBetween(-3, 3), 54, 132)),
    accelX: randomBetween(-0.22, 0.22),
    accelY: randomBetween(-0.18, 0.18),
    accelZ: randomBetween(0.82, 1.18),
  };
};

const simulatePrediction = (samples: TelemetrySample[]): PredictionResult => {
  const sample = samples[samples.length - 1];
  const accelerationMagnitudeG = Math.sqrt(
    sample.accelX * sample.accelX
    + sample.accelY * sample.accelY
    + sample.accelZ * sample.accelZ,
  );
  const predictedCoPpm = clamp(
    sample.coPpm + Math.sin(Date.now() / 30_000) * 8 + randomBetween(3, 18),
    4,
    120,
  );
  const fallSvmPrediction = fallSvmPredictor.predict(samples);
  const tachycardiaRisk = computeTachycardiaRisk(samples);

  return {
    predictedCoPpm,
    horizonSeconds: 600,
    isCriticalCo: predictedCoPpm >= 70,
    isFallDetected: fallSvmPrediction.isFallDetected,
    fallSvmDecisionScore: fallSvmPrediction.decisionScore,
    fallSvmPredictedClass: fallSvmPrediction.predictedClass,
    tachycardiaRiskScore: tachycardiaRisk.score,
    isAcutePhysiologicalDistress: tachycardiaRisk.isAcutePhysiologicalDistress,
    accelerationMagnitudeG,
    timestampMs: Date.now(),
  };
};

const simulateFutureTrend = (currentCo: number): FuturePoint[] => {
  const slope = randomBetween(0.8, 3.8);
  return Array.from({ length: 6 }, (_, index) => {
    const minute = index + 5;
    const coPpm = clamp(currentCo + slope * index + Math.sin(index * 0.9) * 5 + randomBetween(-3, 4), 2, 140);
    return { minute, coPpm };
  });
};

function App() {
  const [latestSample, setLatestSample] = React.useState<TelemetrySample | null>(null);
  const [latestPrediction, setLatestPrediction] = React.useState<PredictionResult | null>(null);
  const [chartSamples, setChartSamples] = React.useState<TelemetrySample[]>(() => createInitialSamples());
  const [futureTrend, setFutureTrend] = React.useState<FuturePoint[]>(() => simulateFutureTrend(12));
  const [connectionState, setConnectionState] = React.useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [isMonitoring, setIsMonitoring] = React.useState(false);
  const [status, setStatus] = React.useState('Idle');
  const [showStartupNotice, setShowStartupNotice] = React.useState(true);

  React.useEffect(() => {
    if (!isMonitoring) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setChartSamples((samples) => {
        const previous = samples.length > 0 ? samples[samples.length - 1] : null;
        const next = simulateNextSample(previous);
        const nextSamples = [...samples.slice(-39), next];
        setLatestSample(next);
        setLatestPrediction(simulatePrediction(nextSamples));
        setFutureTrend(simulateFutureTrend(next.coPpm));
        return nextSamples;
      });
    }, 250);

    return () => window.clearInterval(timer);
  }, [isMonitoring]);

  const start = () => {
    setConnectionState('connecting');
    setStatus('Starting monitor');
    window.setTimeout(() => {
      setConnectionState('connected');
      setIsMonitoring(true);
      setStatus('Simulated monitoring');
    }, 600);
  };

  const stop = () => {
    setConnectionState('disconnected');
    setIsMonitoring(false);
    setStatus('Stopped');
  };

  const coPoints = chartSamples.slice(-40);
  const maxCo = Math.max(100, ...coPoints.map((sample) => sample.coPpm));
  const maxFutureCo = Math.max(100, ...futureTrend.map((point) => point.coPpm));
  const fallDetected = latestPrediction?.isFallDetected ?? false;

  return (
    <main className="app-shell">
      {showStartupNotice && (
        <div className="modal-backdrop" role="presentation">
          <section className="startup-modal" role="dialog" aria-modal="true" aria-labelledby="startup-title">
            <p className="eyebrow">Simulation Mode</p>
            <h2 id="startup-title">CODetect monitor ready</h2>
            <p>
              Live telemetry is simulated for now. Press Start to generate CO,
              heart rate, movement, and forecast values.
            </p>
            <button className="primary" onClick={() => setShowStartupNotice(false)}>
              Continue
            </button>
          </section>
        </div>
      )}

      <section className="status-bar">
        <div>
          <p className="eyebrow">CODetect</p>
          <h1>Live Safety Monitor</h1>
        </div>
        <button className={isMonitoring ? 'secondary' : 'primary'} onClick={isMonitoring ? stop : start}>
          {isMonitoring ? 'Stop' : 'Start'}
        </button>
      </section>

      <section className="metrics-grid">
        <Metric label="CO" value={latestSample ? latestSample.coPpm.toFixed(1) : '--'} unit="ppm" />
        <Metric label="Heart Rate" value={latestSample ? String(latestSample.bpm) : '--'} unit="bpm" />
        <Metric label="Accel X" value={latestSample ? latestSample.accelX.toFixed(2) : '--'} unit="g" />
        <Metric label="Accel Y" value={latestSample ? latestSample.accelY.toFixed(2) : '--'} unit="g" />
        <Metric label="Accel Z" value={latestSample ? latestSample.accelZ.toFixed(2) : '--'} unit="g" />
        <Metric
          label="Forecast"
          value={latestPrediction ? latestPrediction.predictedCoPpm.toFixed(1) : '--'}
          unit="ppm"
        />
        <article className={fallDetected ? 'fall-status detected' : 'fall-status'}>
          <span>Fall Status</span>
          <strong>{fallDetected ? 'DETECTED' : 'None'}</strong>
          <small>{latestPrediction ? `SVM ${latestPrediction.fallSvmDecisionScore.toFixed(2)}` : 'waiting'}</small>
        </article>
        <article className={latestPrediction?.isAcutePhysiologicalDistress ? 'risk-status detected' : 'risk-status'}>
          <span>Tachycardia Risk</span>
          <strong>{latestPrediction ? latestPrediction.tachycardiaRiskScore.toFixed(2) : '--'}</strong>
          <small>{latestPrediction?.isAcutePhysiologicalDistress ? 'distress' : 'threshold 0.70'}</small>
        </article>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>CO Trend</h2>
          <span className={`connection ${connectionState}`}>{connectionState}</span>
        </div>
        <div className="sparkline" aria-label="CO trend">
          {coPoints.map((sample) => (
            <span
              key={`${sample.timestampMs}-${sample.coPpm}`}
              style={{ height: `${Math.max(4, (sample.coPpm / maxCo) * 100)}%` }}
            />
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Predicted CO Trend</h2>
          <span>5-10 min</span>
        </div>
        <div className="future-grid" aria-label="Predicted carbon monoxide trend for the next 5 to 10 minutes">
          {futureTrend.map((point) => (
            <div className="future-point" key={point.minute}>
              <span>{point.minute}m</span>
              <div>
                <i style={{ height: `${Math.max(8, (point.coPpm / maxFutureCo) * 100)}%` }} />
              </div>
              <strong>{point.coPpm.toFixed(0)}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className={latestPrediction?.isCriticalCo || latestPrediction?.isFallDetected || latestPrediction?.isAcutePhysiologicalDistress ? 'alert active' : 'alert'}>
        <strong>{latestPrediction?.isCriticalCo ? 'Critical CO forecast' : latestPrediction?.isFallDetected ? 'Fall risk' : latestPrediction?.isAcutePhysiologicalDistress ? 'Physiological distress' : status}</strong>
        <span>
          {latestPrediction
            ? `${latestPrediction.predictedCoPpm.toFixed(1)} ppm simulated forecast, ${latestPrediction.accelerationMagnitudeG.toFixed(2)} g, R_CT ${latestPrediction.tachycardiaRiskScore.toFixed(2)}`
            : 'Press Start to simulate the CODetect connection.'}
        </span>
      </section>
    </main>
  );
}

function Metric({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <article className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{unit}</small>
    </article>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
