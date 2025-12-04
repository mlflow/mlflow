/**
 * Telemetry module
 *
 * Provides telemetry logging functionality using a SharedWorker
 * to coordinate logging across multiple browser tabs.
 */

export { telemetryClient } from './TelemetryLogger';
export type { TelemetryRecord } from './types';
