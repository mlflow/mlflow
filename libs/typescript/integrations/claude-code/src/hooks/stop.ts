import { readStdin } from '../utils/stdin.js';
import { isTracingEnabled, ensureInitialized } from '../config.js';
import { processTranscript } from '../tracing.js';
import type { StopHookInput } from '../types.js';

async function main(): Promise<void> {
  try {
    const input = await readStdin<StopHookInput>();
    if (!isTracingEnabled()) {
      return;
    }
    if (!ensureInitialized()) {
      return;
    }
    await processTranscript(input.transcript_path, input.session_id);
  } catch (err) {
    console.error('[mlflow]', err);
  }
}

void main();
