import { execFileSync } from 'node:child_process';

const GIT_BRANCH_METADATA_KEY = 'mlflow.source.git.branch';
const GIT_COMMIT_METADATA_KEY = 'mlflow.source.git.commit';
const GIT_REPO_URL_METADATA_KEY = 'mlflow.source.git.repoURL';

type GitCommandRunner = (args: string[], cwd: string) => string | undefined;

let cachedEnvironmentMetadata: Record<string, string> | undefined;

function runGitCommand(args: string[], cwd: string): string | undefined {
  try {
    const output = execFileSync('git', args, {
      cwd,
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 1000,
    }).trim();
    return output || undefined;
  } catch {
    return undefined;
  }
}

function stripCredentialsFromUrl(url: string): string {
  try {
    const parsed = new URL(url);
    if (!parsed.username && !parsed.password) {
      return url;
    }
    parsed.username = '';
    parsed.password = '';
    return parsed.toString();
  } catch {
    // SSH-style remotes such as git@github.com:org/repo.git do not contain
    // URL userinfo credentials and should be preserved as-is.
    return url;
  }
}

export function resolveGitMetadata(
  cwd: string = process.cwd(),
  run: GitCommandRunner = runGitCommand,
): Record<string, string> {
  const commit = run(['rev-parse', 'HEAD'], cwd);
  const branch = run(['branch', '--show-current'], cwd);
  const repoUrl = run(['config', '--get', 'remote.origin.url'], cwd);

  return {
    ...(commit ? { [GIT_COMMIT_METADATA_KEY]: commit } : {}),
    ...(branch ? { [GIT_BRANCH_METADATA_KEY]: branch } : {}),
    ...(repoUrl ? { [GIT_REPO_URL_METADATA_KEY]: stripCredentialsFromUrl(repoUrl) } : {}),
  };
}

export function resolveEnvironmentMetadata(): Record<string, string> {
  cachedEnvironmentMetadata ??= resolveGitMetadata();
  return cachedEnvironmentMetadata;
}
