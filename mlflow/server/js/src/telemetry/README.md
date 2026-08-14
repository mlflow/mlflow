# MLflow UI Telemetry

## Overview

Telemetry in MLflow's UI is based on a [SharedWorker](https://developer.mozilla.org/en-US/docs/Web/API/SharedWorker), which is a type of web worker that can be accessed by multiple tabs. This allows us to effectively consolidate and batch logs.

## Architecture

### Client

**TelemetryClient.ts**:

- Client API for logging events. A singleton is exported from this file, and
  is imported / used in `app.tsx` (the top-level MLflow frontend component).
- We hook into the built-in `DesignSystemEventProvider`, which generates view and click events for every interactive component.
- When `logEvent` is called, the client forwards the log to the SharedWorker via postMessage

### SharedWorker

**TelemetryLogger.worker.ts**:

- Main worker class that handles communication with the `/ui-telemetry` server endpoint.
- Please keep external dependencies here minimal, as it is bundled separately from the main app and we'd ideally keep the generated bundle relatively light (see the `telemetry-worker` entrypoint in `craco.config.js`)

**LogQueue.ts**:

- Simple class that batches logs and uploads them to the server every 15s.
