import { runSetup, runStatus } from './commands/setup.js';

function printUsage(): void {
  console.error('Usage: mlflow-claude-code <command> [options]');
  console.error('');
  console.error('Commands:');
  console.error('  setup        Configure MLflow tracing for Claude Code (all flags required)');
  console.error('               Flags:');
  console.error('                 --project, -p          Write to ./.claude/settings.json');
  console.error('                 --user, -u             Write to ~/.claude/settings.json');
  console.error('                 --tracking-uri <uri>   MLflow tracking URI (required)');
  console.error('                 --experiment-id <id>   Use an existing experiment ID');
  console.error('                 --experiment-name <n>  Create or reuse an experiment by name');
  console.error('');
  console.error('  status       Show the current tracing configuration');
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
    runStatus();
    return;
  }

  console.error(`Unknown command: ${command}\n`);
  printUsage();
  process.exitCode = 1;
}

void main();
