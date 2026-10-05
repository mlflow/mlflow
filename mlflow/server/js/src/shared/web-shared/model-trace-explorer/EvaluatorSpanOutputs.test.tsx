import { describe, expect, it } from '@jest/globals';

import { getEvaluatorOutputFields } from './EvaluatorSpanOutputs';
import { getSpanTokenUsage } from './v2/ModelTraceTokenUsage.utils';

describe('evaluator span outputs', () => {
  it('puts feedback and rationale first and omits null fields', () => {
    expect(
      getEvaluatorOutputFields({
        metadata: { source: 'judge', unused: null },
        error: null,
        rationale: 'Grounded in the retrieved policy',
        feedback: 'grounded',
      }),
    ).toEqual([
      { key: 'feedback', value: '"grounded"' },
      { key: 'rationale', value: '"Grounded in the retrieved policy"' },
      { key: 'metadata', value: '{\n  "source": "judge"\n}' },
    ]);
  });

  it('unwraps feedback from previously stored assessment outputs', () => {
    expect(getEvaluatorOutputFields({ feedback: { value: true, metadata: null }, rationale: 'Correct' })[0]).toEqual({
      key: 'feedback',
      value: 'true',
    });
  });

  it('shows older judge token usage from feedback metadata', () => {
    expect(
      getSpanTokenUsage({
        outputs: {
          metadata: {
            'mlflow.assessment.judgeInputTokens': 125,
            'mlflow.assessment.judgeOutputTokens': 37,
          },
        },
      }),
    ).toEqual({ input_tokens: 125, output_tokens: 37, total_tokens: 162 });
  });
});
