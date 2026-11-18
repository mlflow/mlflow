import { readStdin } from '../utils/stdin.js';
import { isTracingEnabled, ensureInitialized } from '../config.js';
import { isBackgroundSubagent, processSubagentTranscript } from '../subagentTracing.js';
import type { SubagentStopHookInput } from '../types.js';

async function main(): Promise<void> {
  try {
    const input = await readStdin<SubagentStopHookInput>();
    if (!isTracingEnabled()) {
      return;
    }
    // Sync and internal agents are not traced here; skip the tracking-server
    // connection that ensureInitialized would make for them.
    if (!isBackgroundSubagent(input)) {
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
