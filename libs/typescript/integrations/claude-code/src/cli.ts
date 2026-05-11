import { runSetup, runStatus } from './commands/setup.js';
import { FAIL, bold, dim } from './ui.js';

function printUsage(): void {
  console.error(`${bold('Usage:')} mlflow-claude-code <command> [options]`);
  console.error('');
  console.error(bold('Commands:'));
  console.error(`  ${bold('setup')}        Configure MLflow tracing for Claude Code`);
  console.error(`               ${dim('Flags:')}`);
  console.error(
    dim('                 --project, -p          Write to ./.claude/settings.json'),
  );
  console.error(dim('                 --user, -u             Write to ~/.claude/settings.json'));
  console.error(dim('                 --non-interactive, -y  Skip prompts; use flag values or defaults'));
  console.error(dim('                 --tracking-uri <uri>   Set the MLflow tracking URI'));
  console.error(dim('                 --experiment-id <id>   Use an existing experiment ID'));
  console.error(dim('                 --experiment-name <n>  Create or reuse an experiment by name'));
  console.error('');
  console.error(`  ${bold('status')}       Show the current tracing configuration`);
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

  if (command === 'status') {
    await runStatus();
    return;
  }

  console.error(`${FAIL} Unknown command: ${bold(command)}\n`);
  printUsage();
  process.exitCode = 1;
}

void main();
