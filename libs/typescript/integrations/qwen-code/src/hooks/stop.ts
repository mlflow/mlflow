/**
 * Qwen Code Stop hook handler.
 *
 * Qwen Code fires the Stop hook via stdin with JSON:
 *   {"session_id": "...", "transcript_path": "...", "cwd": "...", ...}
 *
 * Configured in .qwen/settings.json under hooks.Stop.
 */

import { readStdin } from '../utils/stdin.js';
import { ensureInitialized } from '../config.js';
import { processTranscript } from '../tracing.js';
import type { StopHookInput } from '../types.js';

export async function runStopHook(): Promise<void> {
  try {
    if (!ensureInitialized()) {
      return;
    }
    const input = await readStdin<StopHookInput>();
    await processTranscript(input.transcript_path, input.session_id);
  } catch (err) {
    console.error('[mlflow]', err);
  }
}
