/**
 * Telemetry module
 *
 * Provides telemetry logging functionality using a SharedWorker
 * to coordinate logging across multiple browser tabs.
 */

export { telemetryClient } from './TelemetryClient';
export type { TelemetryRecord } from './worker/types';
