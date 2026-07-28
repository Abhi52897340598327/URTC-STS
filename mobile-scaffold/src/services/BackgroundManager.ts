import { Capacitor } from '@capacitor/core';
import { LocalNotifications } from '@capacitor/local-notifications';
import {
  ForegroundService,
  Importance,
  ServiceType,
} from '@capawesome-team/capacitor-android-foreground-service';
import type { MlPredictorService } from './MlPredictorService';
import type { WebSocketService } from './WebSocketService';
import type { PredictionResult } from './telemetryTypes';

type BackgroundManagerOptions = {
  foregroundNotificationId?: number;
  foregroundChannelId?: string;
  alertChannelId?: string;
};

/**
 * Coordinates long-running monitoring.
 *
 * Android can keep JavaScript alive only while the WebView process survives.
 * A foreground service plus wake lock materially improves survival, but do not
 * treat it as a hard real-time runtime. The native foreground notification is
 * the user-visible contract that monitoring is active.
 */
export class BackgroundManager {
  private readonly websocket: WebSocketService;
  private readonly ml: MlPredictorService;
  private readonly foregroundNotificationId: number;
  private readonly foregroundChannelId: string;
  private readonly alertChannelId: string;
  private unsubscribeTelemetry: (() => void) | null = null;
  private lastAlertAtMs = 0;

  constructor(
    websocket: WebSocketService,
    ml: MlPredictorService,
    options: BackgroundManagerOptions = {},
  ) {
    this.websocket = websocket;
    this.ml = ml;
    this.foregroundNotificationId = options.foregroundNotificationId ?? 1001;
    this.foregroundChannelId = options.foregroundChannelId ?? 'codetect-monitoring';
    this.alertChannelId = options.alertChannelId ?? 'codetect-critical-alerts';
  }

  async startMonitoring(): Promise<void> {
    await this.prepareNotifications();
    await this.startAndroidForegroundService();
    await this.ml.init();

    this.unsubscribeTelemetry = this.websocket.subscribe(async (sample) => {
      const prediction = await this.ml.ingestLiveSample(sample);
      if (prediction?.isCriticalCo || prediction?.isFallDetected) {
        await this.raiseCriticalAlert(prediction);
      }
    });

    this.websocket.connect();
  }

  async stopMonitoring(): Promise<void> {
    this.unsubscribeTelemetry?.();
    this.unsubscribeTelemetry = null;
    this.websocket.disconnect();

    if (Capacitor.getPlatform() === 'android') {
      await ForegroundService.stopForegroundService();
    }
  }

  private async prepareNotifications(): Promise<void> {
    const permissions = await LocalNotifications.checkPermissions();
    if (permissions.display !== 'granted') {
      await LocalNotifications.requestPermissions();
    }

    if (Capacitor.getPlatform() === 'android') {
      await ForegroundService.createNotificationChannel({
        id: this.foregroundChannelId,
        name: 'CODetect monitoring',
        description: 'Keeps CODetect monitoring active in the background.',
        importance: Importance.Default,
      });

      await LocalNotifications.createChannel({
        id: this.alertChannelId,
        name: 'CODetect critical alerts',
        description: 'Critical carbon monoxide and fall alerts.',
        importance: 5,
        visibility: 1,
        sound: 'default',
        vibration: true,
        lights: true,
      });
    }
  }

  private async startAndroidForegroundService(): Promise<void> {
    if (Capacitor.getPlatform() !== 'android') {
      return;
    }

    await ForegroundService.requestPermissions();
    await ForegroundService.startForegroundService({
      id: this.foregroundNotificationId,
      title: 'CODetect monitoring',
      body: 'Watching CO, heart rate, and movement.',
      smallIcon: 'ic_stat_codetect',
      notificationChannelId: this.foregroundChannelId,
      silent: false,
      // The current Capawesome docs expose ServiceType. Cast to string fallback
      // so older enum typings do not block using Android 14's connectedDevice.
      serviceType: (ServiceType as Record<string, string>).ConnectedDevice ?? 'connectedDevice',
    });
  }

  private async raiseCriticalAlert(prediction: PredictionResult): Promise<void> {
    const now = Date.now();
    if (now - this.lastAlertAtMs < 15_000) {
      return;
    }
    this.lastAlertAtMs = now;

    const title = prediction.isCriticalCo
      ? 'Critical CO risk'
      : 'Possible fall detected';
    const body = prediction.isCriticalCo
      ? `Predicted CO ${prediction.predictedCoPpm.toFixed(0)} ppm in ${prediction.horizonSeconds.toFixed(0)}s.`
      : `Acceleration spike ${prediction.accelerationMagnitudeG.toFixed(1)} g detected.`;

    await LocalNotifications.schedule({
      notifications: [{
        id: now % 2_147_483_647,
        title,
        body,
        channelId: this.alertChannelId,
        schedule: { at: new Date(now + 100) },
        ongoing: false,
        autoCancel: false,
        extra: {
          predictedCoPpm: prediction.predictedCoPpm,
          accelerationMagnitudeG: prediction.accelerationMagnitudeG,
        },
      }],
    });

    if (Capacitor.getPlatform() === 'android') {
      await ForegroundService.updateForegroundService({
        id: this.foregroundNotificationId,
        title,
        body,
        smallIcon: 'ic_stat_codetect',
        notificationChannelId: this.foregroundChannelId,
        silent: false,
      });

      // Brings the app forward only if overlay permission is granted. Request
      // this during onboarding because Android shows a special settings screen.
      const overlay = await ForegroundService.checkManageOverlayPermission();
      if (overlay.granted) {
        await ForegroundService.moveToForeground();
      }
    }
  }
}
