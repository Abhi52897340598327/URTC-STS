import React from 'react';
import ReactDOM from 'react-dom/client';
import { useStore } from 'zustand';
import './styles.css';
import { BackgroundManager } from './services/BackgroundManager';
import { MlPredictorService } from './services/MlPredictorService';
import { WebSocketService } from './services/WebSocketService';
import { createThrottledUiPublisher, telemetryStore } from './services/telemetryStore';

const websocket = new WebSocketService();
const ml = new MlPredictorService();
const background = new BackgroundManager(websocket, ml);
const publishUiSample = createThrottledUiPublisher(250);

websocket.subscribe((sample) => publishUiSample(sample));
ml.subscribe((prediction) => telemetryStore.getState().publishPrediction(prediction));

function App() {
  const latestSample = useStore(telemetryStore, (state) => state.latestSample);
  const latestPrediction = useStore(telemetryStore, (state) => state.latestPrediction);
  const chartSamples = useStore(telemetryStore, (state) => state.chartSamples);
  const connectionState = useStore(telemetryStore, (state) => state.connectionState);
  const [isMonitoring, setIsMonitoring] = React.useState(false);
  const [status, setStatus] = React.useState('Idle');

  const start = async () => {
    telemetryStore.getState().setConnectionState('connecting');
    setStatus('Starting monitor');
    await background.startMonitoring();
    telemetryStore.getState().setConnectionState('connected');
    setIsMonitoring(true);
    setStatus('Monitoring');
  };

  const stop = async () => {
    await background.stopMonitoring();
    telemetryStore.getState().setConnectionState('disconnected');
    setIsMonitoring(false);
    setStatus('Stopped');
  };

  const coPoints = chartSamples.slice(-40);
  const maxCo = Math.max(100, ...coPoints.map((sample) => sample.coPpm));

  return (
    <main className="app-shell">
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
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>CO Trend</h2>
          <span>{connectionState}</span>
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

      <section className={latestPrediction?.isCriticalCo || latestPrediction?.isFallDetected ? 'alert active' : 'alert'}>
        <strong>{latestPrediction?.isCriticalCo ? 'Critical CO forecast' : latestPrediction?.isFallDetected ? 'Fall risk' : status}</strong>
        <span>
          {latestPrediction
            ? `${latestPrediction.predictedCoPpm.toFixed(1)} ppm forecast, ${latestPrediction.accelerationMagnitudeG.toFixed(2)} g`
            : 'Connect to the ESP32 SoftAP, then start monitoring.'}
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
