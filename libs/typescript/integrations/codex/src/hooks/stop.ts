/**
 * Codex notify hook entry point.
 *
 * Codex passes the turn data as a JSON string in the first CLI argument:
 *   node stop.js '{"type":"agent-turn-complete","thread-id":"...","input-messages":[...],...}'
 *
 * Configured in ~/.codex/config.toml:
 *   notify = ["node", "/path/to/bundle/stop.js"]
 */

import { isTracingEnabled, ensureInitialized } from '../config.js';
import { processNotify } from '../tracing.js';
import type { NotifyPayload } from '../types.js';

async function main(): Promise<void> {
  try {
    const arg = process.argv[2];
    if (!arg) {
      return;
    }

    const payload = JSON.parse(arg) as NotifyPayload;
    if (payload.type !== 'agent-turn-complete') {
      return;
    }

    if (!isTracingEnabled()) {
      return;
    }
    if (!ensureInitialized()) {
      return;
    }

    await processNotify(payload);
  } catch (err) {
    console.error('[mlflow]', err);
  }
}

void main();
