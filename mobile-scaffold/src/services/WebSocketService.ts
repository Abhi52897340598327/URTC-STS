import type { TelemetryListener, TelemetrySample } from './telemetryTypes';

type WebSocketServiceOptions = {
  url?: string;
  maxBackoffMs?: number;
  baseBackoffMs?: number;
  maxBufferedSamples?: number;
};

const DEFAULT_URL = 'ws://192.168.4.1/ws';
const BINARY_PACKET_BYTES = 18;

/**
 * Owns the ESP32 SoftAP websocket connection.
 *
 * This class intentionally does not write every packet into React state.
 * At 10 Hz that causes needless reconciliation work and can freeze low-end
 * Android devices when TensorFlow is also active. UI code should subscribe and
 * throttle its own rendering, while ML/background services consume every sample.
 */
export class WebSocketService {
  private readonly url: string;
  private readonly baseBackoffMs: number;
  private readonly maxBackoffMs: number;
  private readonly maxBufferedSamples: number;

  private socket: WebSocket | null = null;
  private listeners = new Set<TelemetryListener>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempt = 0;
  private manuallyClosed = false;
  private buffer: TelemetrySample[] = [];

  constructor(options: WebSocketServiceOptions = {}) {
    this.url = options.url ?? DEFAULT_URL;
    this.baseBackoffMs = options.baseBackoffMs ?? 500;
    this.maxBackoffMs = options.maxBackoffMs ?? 30_000;
    this.maxBufferedSamples = options.maxBufferedSamples ?? 60 * 10;
  }

  connect(): void {
    if (this.socket?.readyState === WebSocket.OPEN || this.socket?.readyState === WebSocket.CONNECTING) {
      return;
    }

    this.manuallyClosed = false;
    this.clearReconnectTimer();

    const socket = new WebSocket(this.url);
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempt = 0;
    };

    socket.onmessage = (event: MessageEvent<ArrayBuffer | Blob>) => {
      this.handleMessage(event.data);
    };

    socket.onerror = () => {
      // The close event normally follows. If it does not, close explicitly so
      // the reconnect path runs from one place.
      if (socket.readyState !== WebSocket.CLOSING && socket.readyState !== WebSocket.CLOSED) {
        socket.close();
      }
    };

    socket.onclose = () => {
      if (this.socket === socket) {
        this.socket = null;
      }
      if (!this.manuallyClosed) {
        this.scheduleReconnect();
      }
    };
  }

  disconnect(): void {
    this.manuallyClosed = true;
    this.clearReconnectTimer();

    if (this.socket) {
      this.socket.onopen = null;
      this.socket.onmessage = null;
      this.socket.onerror = null;
      this.socket.onclose = null;
      this.socket.close();
      this.socket = null;
    }
  }

  subscribe(listener: TelemetryListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getLatestSamples(limit = this.maxBufferedSamples): TelemetrySample[] {
    return this.buffer.slice(-limit);
  }

  isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  private handleMessage(data: ArrayBuffer | Blob): void {
    if (data instanceof ArrayBuffer) {
      this.emitSample(WebSocketService.parseTelemetryPacket(data));
      return;
    }

    // Blob conversion allocates, but this path should only happen on browsers
    // that ignore binaryType. The ESP32 payload remains the same 18 bytes.
    data.arrayBuffer()
      .then((buffer) => this.emitSample(WebSocketService.parseTelemetryPacket(buffer)))
      .catch((error) => console.error('Failed to read telemetry blob', error));
  }

  private emitSample(sample: TelemetrySample): void {
    this.buffer.push(sample);
    if (this.buffer.length > this.maxBufferedSamples) {
      this.buffer.splice(0, this.buffer.length - this.maxBufferedSamples);
    }

    for (const listener of this.listeners) {
      listener(sample);
    }
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer();

    const exponentialDelay = this.baseBackoffMs * 2 ** this.reconnectAttempt;
    const cappedDelay = Math.min(exponentialDelay, this.maxBackoffMs);
    const jitter = Math.round(cappedDelay * 0.2 * Math.random());
    const delay = cappedDelay + jitter;

    this.reconnectAttempt += 1;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  static parseTelemetryPacket(buffer: ArrayBuffer): TelemetrySample {
    if (buffer.byteLength !== BINARY_PACKET_BYTES) {
      throw new Error(`Invalid telemetry packet size: expected ${BINARY_PACKET_BYTES}, got ${buffer.byteLength}`);
    }

    const view = new DataView(buffer);

    return {
      timestampMs: Date.now(),
      // Byte 0-3: CO_PPM, Float32, little endian.
      coPpm: view.getFloat32(0, true),
      // Byte 4-5: BPM, Uint16, little endian.
      bpm: view.getUint16(4, true),
      // Byte 6-9: Accel_X, Float32, little endian.
      accelX: view.getFloat32(6, true),
      // Byte 10-13: Accel_Y, Float32, little endian.
      accelY: view.getFloat32(10, true),
      // Byte 14-17: Accel_Z, Float32, little endian.
      accelZ: view.getFloat32(14, true),
    };
  }
}
