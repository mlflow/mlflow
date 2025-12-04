# TelemetryLogger

A SharedWorker-based telemetry logging system for the MLflow UI.

## Overview

The TelemetryLogger uses a SharedWorker to coordinate telemetry logging across multiple browser tabs/windows. This ensures that:

- Telemetry events are logged efficiently without duplication
- Configuration is fetched once and shared across all tabs
- Events are filtered based on server-side configuration

## Architecture

- **TelemetryLogger.ts**: Client API for logging events
- **TelemetryLogger.worker.ts**: SharedWorker that handles logging and config
- **types.ts**: TypeScript types for telemetry data

## Usage

### Basic Event Logging

```typescript
import { getTelemetryLogger, TelemetryStatus } from '@/telemetry';

const logger = getTelemetryLogger();

// Log a simple event
logger.logEvent({
  event_name: 'button_clicked',
});

// Log an event with parameters
logger.logEvent({
  event_name: 'experiment_created',
  params: {
    experiment_id: '123',
    source: 'ui',
  },
  status: TelemetryStatus.SUCCESS,
});

// Log an event with timing
logger.logEvent({
  event_name: 'page_loaded',
  duration_ms: 1250,
  status: TelemetryStatus.SUCCESS,
});
```

### Get Configuration

```typescript
import { getTelemetryLogger } from '@/telemetry';

const logger = getTelemetryLogger();
const config = await logger.getConfig();

if (config) {
  console.log('Ingestion URL:', config.ingestion_url);
  console.log('Disabled events:', config.disable_events);
}
```

### Check Initialization Status

```typescript
import { getTelemetryLogger } from '@/telemetry';

const logger = getTelemetryLogger();
if (logger.isInitialized()) {
  logger.logEvent({ event_name: 'app_started' });
}
```

## Event Record Structure

```typescript
interface TelemetryRecord {
  event_name: string; // Required: Name of the event
  timestamp_ns: number; // Optional: Defaults to current time
  params?: Record<string, any>; // Optional: Event parameters
  status?: TelemetryStatus; // Optional: Event status (UNKNOWN, SUCCESS, FAILURE)
  duration_ms?: number; // Optional: Event duration in milliseconds
}
```

## How It Works

1. **Initialization**: When first imported, TelemetryLogger creates a SharedWorker
2. **Config Fetch**: The worker fetches telemetry config from `/ajax-api/2.0/mlflow/telemetry` (GET)
3. **Event Logging**: Events are sent to the worker, which filters and posts them to the server
4. **Caching**: Config is cached in the worker to avoid repeated fetches

## Browser Support

SharedWorker is supported in:
- Chrome/Edge 4+
- Firefox 29+
- Safari 16+

If SharedWorker is not supported, the logger will fail gracefully and log a warning.

## Notes

- Events are fire-and-forget; `logEvent()` does not return a Promise
- The logger automatically filters disabled events based on server config
- Timestamps are automatically added if not provided (in nanoseconds)
- The SharedWorker is shared across all tabs from the same origin
