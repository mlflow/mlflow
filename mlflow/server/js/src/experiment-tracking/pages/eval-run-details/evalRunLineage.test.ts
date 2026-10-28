import { describe, expect, test } from '@jest/globals';
import {
  EVAL_RUN_AGENT_DIGEST_TAG,
  EVAL_RUN_AGENT_FUNCTION_TAG,
  EVAL_RUN_AGENT_URI_TAG,
  formatShortDigest,
  getEvalRunAgentLineage,
  getEvalRunGitCommit,
  parseEvalRunScorersDigest,
} from './evalRunLineage';

describe('parseEvalRunScorersDigest', () => {
  test.each([
    [
      { digest: 'sha256:a', hashed: 2, total: 2 },
      { digest: 'sha256:a', coverage: 'complete' },
    ],
    [
      { digest: 'sha256:a', hashed: 1, total: 2 },
      { digest: 'sha256:a', coverage: 'partial' },
    ],
    [
      { hashed: 0, total: 2 },
      { digest: undefined, coverage: 'unavailable' },
    ],
  ])('reads %j', (value, expected) => {
    expect(parseEvalRunScorersDigest(JSON.stringify(value))).toEqual({
      hashed: value.hashed,
      total: value.total,
      ...expected,
    });
  });

  test.each([
    ['missing tag', undefined],
    ['invalid JSON', '{'],
    ['no scorers', JSON.stringify({ hashed: 0, total: 0 })],
    ['more hashed than total', JSON.stringify({ digest: 'sha256:a', hashed: 3, total: 2 })],
    ['hashed scorers without a digest', JSON.stringify({ hashed: 1, total: 1 })],
    ['non-integer counts', JSON.stringify({ digest: 'sha256:a', hashed: 1.5, total: 2 })],
  ])('ignores %s', (_name, value) => {
    expect(parseEvalRunScorersDigest(value)).toBeUndefined();
  });
});

describe('formatShortDigest', () => {
  test('keeps the first 7 hex characters after the algorithm prefix', () => {
    expect(formatShortDigest('sha256:3f9a1c2d4e5f')).toBe('3f9a1c2');
    expect(formatShortDigest('abcdef0123')).toBe('abcdef0');
  });
});

describe('getEvalRunAgentLineage', () => {
  test('reads the agent tags and the logged model', () => {
    expect(
      getEvalRunAgentLineage(
        {
          [EVAL_RUN_AGENT_FUNCTION_TAG]: 'my_pkg.agent.predict',
          [EVAL_RUN_AGENT_URI_TAG]: 'endpoints:/support-agent',
          [EVAL_RUN_AGENT_DIGEST_TAG]: 'sha256:abc',
          'mlflow.source.git.commit': 'local-commit',
        },
        'm-123',
      ),
    ).toEqual({
      function: 'my_pkg.agent.predict',
      uri: 'endpoints:/support-agent',
      digest: 'sha256:abc',
      loggedModelId: 'm-123',
    });
  });

  test('returns undefined when the run has no agent lineage', () => {
    expect(getEvalRunAgentLineage({ 'mlflow.runName': 'run' }, '')).toBeUndefined();
  });

  test('does not treat a git commit alone as agent lineage', () => {
    expect(getEvalRunAgentLineage({ 'mlflow.source.git.commit': 'local-commit' })).toBeUndefined();
  });

  test('treats a logged model alone as agent lineage', () => {
    expect(getEvalRunAgentLineage({}, 'm-123')).toMatchObject({ loggedModelId: 'm-123' });
  });
});

describe('getEvalRunGitCommit', () => {
  test('reads the git commit tag', () => {
    expect(getEvalRunGitCommit({ 'mlflow.source.git.commit': 'local-commit' })).toBe('local-commit');
  });

  test('returns undefined without a commit', () => {
    expect(getEvalRunGitCommit({ 'mlflow.source.git.commit': '' })).toBeUndefined();
    expect(getEvalRunGitCommit({})).toBeUndefined();
  });
});
