# CODetect Mobile Architecture

## Core Dependencies

Use the currently documented Capawesome package name:

```bash
npm install @tensorflow/tfjs zustand @capacitor/local-notifications @capawesome-team/capacitor-android-foreground-service
npx cap sync android
```

The package name in some older examples is easy to mistype. Current Capawesome docs show `@capawesome-team/capacitor-android-foreground-service`.

## AndroidManifest.xml

For Android 14+, foreground seryvices must declare both a service type and the matching type permission. CODetect is a user-visible continuous external-device monitor over local Wi-Fi, so `connectedDevice` is the closest fit. If Play policy review rejects that framing, use `dataSync` only if you position the service as continuous local/network data transfer and processing.

```xml
<manifest>
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_CONNECTED_DEVICE" />
    <!-- Optional alternative if you use android:foregroundServiceType="dataSync". -->
    <!-- <uses-permission android:name="android.permission.FOREGROUND_SERVICE_DATA_SYNC" /> -->
    <uses-permission android:name="android.permission.WAKE_LOCK" />
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
    <uses-permission android:name="android.permission.USE_FULL_SCREEN_INTENT" />
    <uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />

    <application>
        <receiver
            android:name="io.capawesome.capacitorjs.plugins.foregroundservice.NotificationActionBroadcastReceiver" />

        <service
            android:name="io.capawesome.capacitorjs.plugins.foregroundservice.AndroidForegroundService"
            android:exported="false"
            android:foregroundServiceType="connectedDevice" />
    </application>
</manifest>
```

`USE_FULL_SCREEN_INTENT` is restricted on Android 14+. High-importance notifications are still required, but true screen-waking full-screen behavior depends on user/system grant and Play policy. `BackgroundManager` also calls `moveToForeground()` when overlay permission is granted.

## Telemetry Ingestion

`WebSocketService` connects to `ws://192.168.4.1/ws`, sets `binaryType = 'arraybuffer'`, and parses exactly 18 bytes:

```ts
coPpm = view.getFloat32(0, true);
bpm = view.getUint16(4, true);
accelX = view.getFloat32(6, true);
accelY = view.getFloat32(10, true);
accelZ = view.getFloat32(14, true);
```

The service keeps only a bounded in-memory buffer. It emits samples to subscribers without writing each 10 Hz packet into React state.

## GRU Tensor Shaping

Each live sample is five features:

```ts
[coPpm, bpm, accelX, accelY, accelZ]
```

With `timeSteps = 30`, one model input window is 30 rows covering about 3 seconds at 10 Hz. For `N` historical samples and `forecastHorizonSteps = 30`, training creates:

```text
sampleCount = N - timeSteps - forecastHorizonSteps
xs shape    = [sampleCount, 30, 5]
ys shape    = [sampleCount, 1]
```

For sample `i`, `xs[i]` is rows `i..i+29`, and `ys[i]` is the CO PPM at row `i+60`. That trains a 3-second-ahead CO forecast.

The included CSV adapter can train from `data/co-data` CO-only logs by filling BPM and accelerometer features with zeros. That is useful for CO trajectory pretraining, but the production model should be retrained with synchronized wearable logs so the GRU can learn cross-feature patterns.

## Memory and UI Survival

The 10 Hz inference loop uses `tf.tidy()` around tensor creation and prediction, converts the output to a plain number with `dataSync()[0]`, and lets TensorFlow dispose every intermediate tensor immediately. Training tensors are explicitly disposed in a `finally` block after `model.fit()`.

React should consume throttled derived state only, for example updating charts 2-4 times per second from `WebSocketService.getLatestSamples()`. `telemetryStore.ts` is a Zustand vanilla store with `createThrottledUiPublisher(250)`, so service code can process all 10 packets per second while React renders at about 4 Hz.

## Runtime Wiring

```ts
const ws = new WebSocketService({ url: 'ws://192.168.4.1/ws' });
const ml = new MlPredictorService({
  timeSteps: 30,
  forecastHorizonSteps: 30,
  criticalCoPpm: 70,
});
const background = new BackgroundManager(ws, ml);

await background.startMonitoring();
```
