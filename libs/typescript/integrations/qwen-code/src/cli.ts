/**
 * CLI dispatcher for `@mlflow/qwen-code`.
 *
 * Installed as the `mlflow-qwen-code` bin. Subcommands:
 *   - `setup`       → registers the hook and writes mlflow-tracing.json
 *   - `stop-hook`   → runs the Stop hook (reads stdin JSON from Qwen Code)
 *   - `--help`/`-h` → prints usage
 *
 * Qwen Code invokes `mlflow-qwen-code stop-hook` as the registered Stop hook.
 */

import { runStopHook } from './hooks/stop.js';
import { runSetup } from './commands/setup.js';
import { FAIL, bold, dim } from './ui.js';

function printUsage(): void {
  console.error(`${bold('Usage:')} mlflow-qwen-code <command> [options]`);
  console.error('');
  console.error(bold('Commands:'));
  console.error(
    `  ${bold('setup')}       Register the Stop hook in Qwen settings.json and configure`,
  );
  console.error('              the MLflow tracking URI / experiment ID. Runs interactively');
  console.error('              by default.');
  console.error('');
  console.error(`              ${dim('Flags:')}`);
  console.error(
    dim(
      '                --project, -p          Write to ./.qwen/ (skip the interactive scope prompt)',
    ),
  );
  console.error(
    dim('                --non-interactive, -y  Skip prompts; use flag values or defaults'),
  );
  console.error(
    dim('                --tracking-uri <url>   Bypass the prompt for the tracking URI'),
  );
  console.error(
    dim('                --experiment-id <id>   Bypass the prompt for the experiment ID'),
  );
  console.error('');
  console.error(
    `  ${bold('stop-hook')}   Run the Qwen Code Stop hook. Reads the hook payload from stdin`,
  );
  console.error('              — this is the form Qwen itself invokes via settings.json.');
}

async function main(): Promise<void> {
  const [, , command, ...rest] = process.argv;

  if (command === undefined || command === '--help' || command === '-h' || command === 'help') {
    printUsage();
    if (command === undefined) {
      process.exitCode = 1;
    }
    return;
  }

  if (command === 'setup') {
    await runSetup(rest);
    return;
  }

  if (command === 'stop-hook') {
    await runStopHook();
    return;
  }

  console.error(`${FAIL} Unknown command: ${bold(command)}\n`);
  printUsage();
  process.exitCode = 1;
}

void main();
