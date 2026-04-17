/**
 * Codex notify hook entry point.
 *
 * Codex passes the turn data as a JSON string in the first CLI argument:
 *   node stop.js '{"type":"agent-turn-complete","thread-id":"...","input-messages":[...],...}'
 *
 * Configured in ~/.codex/config.toml:
 *   notify = ["node", "/path/to/bundle/stop.js"]
 */

import { ensureInitialized } from '../config.js';
import { processNotify } from '../tracing.js';
import type { NotifyPayload } from '../types.js';

async function main(): Promise<void> {
  try {
    // Initialize early to fail fast if MLFLOW_TRACKING_URI is not set,
    // before spending time parsing the payload.
    if (!ensureInitialized()) {
      return;
    }

    const arg = process.argv[2];
    if (!arg) {
      return;
    }

    const payload = JSON.parse(arg) as NotifyPayload;
    if (payload.type !== 'agent-turn-complete') {
      return;
    }

    await processNotify(payload);
  } catch (err) {
    console.error('[mlflow]', err);
  }
}

void main();
