import { resolveGitMetadata } from '../../../src/core/utils/environment';

describe('resolveGitMetadata', () => {
  it('uses the standard MLflow Git metadata keys', () => {
    const values = new Map([
      ['rev-parse HEAD', 'abc123'],
      ['branch --show-current', 'main'],
      ['config --get remote.origin.url', 'git@github.com:mlflow/mlflow.git'],
    ]);

    expect(resolveGitMetadata('/repo', (args) => values.get(args.join(' ')))).toEqual({
      'mlflow.source.git.branch': 'main',
      'mlflow.source.git.commit': 'abc123',
      'mlflow.source.git.repoURL': 'git@github.com:mlflow/mlflow.git',
    });
  });

  it('returns no metadata outside a Git repository', () => {
    expect(resolveGitMetadata('/not-a-repo', () => undefined)).toEqual({});
  });

  it('keeps available branch and remote metadata in a repository without commits', () => {
    const values = new Map([
      ['branch --show-current', 'main'],
      ['config --get remote.origin.url', 'git@github.com:mlflow/mlflow.git'],
    ]);

    expect(resolveGitMetadata('/repo', (args) => values.get(args.join(' ')))).toEqual({
      'mlflow.source.git.branch': 'main',
      'mlflow.source.git.repoURL': 'git@github.com:mlflow/mlflow.git',
    });
  });

  it('omits unavailable branch and remote values', () => {
    expect(
      resolveGitMetadata('/repo', (args) =>
        args.join(' ') === 'rev-parse HEAD' ? 'abc123' : undefined,
      ),
    ).toEqual({
      'mlflow.source.git.commit': 'abc123',
    });
  });

  it('strips credentials from HTTP remote URLs', () => {
    const values = new Map([
      ['rev-parse HEAD', 'abc123'],
      ['config --get remote.origin.url', 'https://user:secret@example.com/org/repo.git'],
    ]);

    expect(resolveGitMetadata('/repo', (args) => values.get(args.join(' ')))).toMatchObject({
      'mlflow.source.git.repoURL': 'https://example.com/org/repo.git',
    });
  });
});
