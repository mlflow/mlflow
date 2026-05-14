import { runSetup, runStatus } from './commands/setup.js';

function printUsage(): void {
  console.error('Usage: mlflow-claude-code <command> [options]');
  console.error('');
  console.error('Commands:');
  console.error('  setup        Configure MLflow tracing for Claude Code.');
  console.error('               Options:');
  console.error(
    '                 -p, --project          Write settings to ./.claude/settings.json',
  );
  console.error('                                        (this repo only).');
  console.error(
    '                 -u, --user             Write settings to ~/.claude/settings.json',
  );
  console.error('                                        (all repos).');
  console.error('                 --tracking-uri <uri>   MLflow tracking URI. One of');
  console.error("                                        'databricks',");
  console.error("                                        'databricks://<profile>',");
  console.error('                                        or an absolute http(s) URL.');
  console.error('                 --experiment-id <id>   Use an existing MLflow experiment by ID.');
  console.error(
    '                 --experiment-name <n>  Create or reuse an MLflow experiment by name.',
  );
  console.error('');
  console.error('               Required values (all must come from the user; do not pick');
  console.error('               defaults silently):');
  console.error('                 - Scope: pass exactly one of --project or --user.');
  console.error('                 - Tracking URI: pass --tracking-uri.');
  console.error('                 - Experiment: pass exactly one of --experiment-id or');
  console.error('                   --experiment-name.');
  console.error('');
  console.error('               Examples:');
  console.error(
    '                 $ mlflow-claude-code setup --project --tracking-uri http://localhost:5000 \\',
  );
  console.error('                     --experiment-name my-exp');
  console.error(
    '                 $ mlflow-claude-code setup --user --tracking-uri databricks --experiment-id 12345',
  );
  console.error('');
  console.error('  status       Show the current MLflow tracing configuration.');
}

async function main(): Promise<void> {
  const [, , command, ...rest] = process.argv;
  const wantsHelp = rest.includes('--help') || rest.includes('-h');

  if (command === undefined || command === '--help' || command === '-h' || command === 'help') {
    printUsage();
    if (command === undefined) {
      process.exitCode = 1;
    }
    return;
  }

  if (command === 'setup') {
    if (wantsHelp) {
      printUsage();
      return;
    }
    await runSetup(rest);
    return;
  }

  if (command === 'status') {
    if (wantsHelp) {
      printUsage();
      return;
    }
    runStatus();
    return;
  }

  console.error(`Unknown command: ${command}\n`);
  printUsage();
  process.exitCode = 1;
}

void main();
