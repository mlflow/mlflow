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

  it('translates non-System-One endpoint errors into user actions', () => {
    expect(formatJudgeModelError(new Error('Gateway endpoint does not use a System One model.'))).toContain(
      'does not use a TypeSafe or OpenRouter Jev decision model',
    );
  });

  it('translates trace-based rejection into a rewrite action', () => {
    expect(formatJudgeModelError(new Error('TypeSafe judge models do not support trace-based evaluation.'))).toContain(
      'does not support the full {{ trace }} variable',
    );
  });

  it('translates unsupported output type into a Boolean/categorical action', () => {
    expect(
      formatJudgeModelError(
        new Error('TypeSafe judge models support bool or finite Literal feedback value types. ' + "Got <class 'str'>."),
      ),
    ).toContain('only supports Boolean or a fixed list of categorical outcomes');
  });

  it('matches the raw "Failed to invoke judge model" wrapper surfaced on traces', () => {
    expect(
      formatJudgeModelError(
        new Error(
          'Failed to invoke judge model: {"detail":"System One models only support structured ' +
            'evaluation through /gateway/typesafe/v1/systemone. Use a chat model for chat ' +
            'completions."}',
        ),
      ),
    ).toContain('This model is decision-only');
  });

  it('returns null for unrelated errors', () => {
    expect(formatJudgeModelError(new Error('Network request failed'))).toBeNull();
  });
});
