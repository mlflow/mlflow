/**
 * Qwen Code Stop hook entry point.
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

async function main(): Promise<void> {
  try {
    // Initialize early to fail fast if MLFLOW_TRACKING_URI is not set
    if (!ensureInitialized()) {
      return;
    }
    const input = await readStdin<StopHookInput>();
    await processTranscript(input.transcript_path, input.session_id);
  } catch (err) {
    console.error('[mlflow]', err);
  }
}

void main();
