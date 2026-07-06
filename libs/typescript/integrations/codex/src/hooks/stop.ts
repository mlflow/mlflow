/**
 * Codex notify hook handler.
 *
 * Codex passes the turn data as a JSON string in the first CLI argument:
 *   node cli.js '{"type":"agent-turn-complete","thread-id":"...","input-messages":[...],...}'
 *
 * Configured in ~/.codex/config.toml:
 *   notify = ["mlflow-codex"]
 */

import { ensureInitialized } from '../config.js';
import { processNotify } from '../tracing.js';
import type { NotifyPayload } from '../types.js';

export async function runNotifyHook(rawPayload: string): Promise<void> {
  try {
    if (!ensureInitialized()) {
      return;
    }

    const payload = JSON.parse(rawPayload) as NotifyPayload;
    if (payload.type !== 'agent-turn-complete') {
      return;
    }

    await processNotify(payload);
  } catch (err) {
    console.error('[mlflow]', err);
  }
}
