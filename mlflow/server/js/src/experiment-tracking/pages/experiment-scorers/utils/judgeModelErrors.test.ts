import { describe, expect, it } from '@jest/globals';
import { formatJudgeModelError } from './judgeModelErrors';

describe('formatJudgeModelError', () => {
  it('translates backend route errors into user actions', () => {
    expect(
      formatJudgeModelError(
        new Error(
          'System One models only support structured evaluation through /gateway/typesafe/v1/systemone. ' +
            'Use a chat model for chat completions.',
        ),
      ),
    ).toContain('This model is decision-only');
  });

  it('translates mixed-endpoint errors into user actions', () => {
    expect(
      formatJudgeModelError(
        new Error(
          'System One endpoints cannot mix System One and chat models. ' +
            'Every primary and fallback model must support System One.',
        ),
      ),
    ).toContain('This endpoint mixes decision-only and chat models');
  });

  it('returns null for unrelated errors', () => {
    expect(formatJudgeModelError(new Error('Network request failed'))).toBeNull();
  });
});
