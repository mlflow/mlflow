import { readStdin } from '../utils/stdin.js';
import { isTracingEnabled, ensureInitialized } from '../config.js';
import { processSubagentTranscript } from '../subagentTracing.js';
import type { SubagentStopHookInput } from '../types.js';

async function main(): Promise<void> {
  try {
    const input = await readStdin<SubagentStopHookInput>();
    if (!isTracingEnabled()) {
      return;
    }
    if (!(await ensureInitialized())) {
      return;
    }
    await processSubagentTranscript(input);
  } catch (err) {
    console.error('[mlflow]', err);
  }
}

void main();
