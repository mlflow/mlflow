import { describe, expect, it } from '@jest/globals';

import { resolveDecisionViewModel } from './resolveDecisionViewModel';

describe('resolveDecisionViewModel', () => {
  it('returns null when no registered decision translator matches', () => {
    expect(resolveDecisionViewModel()).toBeNull();
    expect(
      resolveDecisionViewModel({
        chatMessageFormat: 'openai',
        inputs: { questions: { relevant: { type: 'noul' } } },
        outputs: { answers: { relevant: { type: 'noul', noul: 0.9 } } },
      }),
    ).toBeNull();
  });

  it('does not infer a decision format from the span title or model provider', () => {
    const span = {
      title: 'typesafe.system_one',
      modelProvider: 'typesafe',
      inputs: { questions: { relevant: { type: 'noul' } } },
      outputs: { answers: { relevant: { type: 'noul', noul: 0.9 } } },
    };

    expect(resolveDecisionViewModel(span)).toBeNull();
  });

  it('uses the matching translator to build a provider-neutral view model', () => {
    expect(
      resolveDecisionViewModel({
        chatMessageFormat: 'typesafe',
        inputs: {
          state: 'hello',
          questions: { relevant: { type: 'noul', instructions: 'Is this relevant?' } },
        },
        outputs: { answers: { relevant: { type: 'noul', noul: 0.9 } } },
      }),
    ).toMatchObject({
      inputs: {
        fields: [
          { kind: 'field', fieldKey: 'state' },
          {
            kind: 'questions',
            fieldKey: 'questions',
            items: [{ id: 'relevant', declaredType: 'noul' }],
          },
        ],
      },
      answers: [{ id: 'relevant', kind: 'noul', probabilityTrue: 0.9 }],
    });
  });

  it('returns null when a matching integration emits a custom response without standard answers', () => {
    expect(
      resolveDecisionViewModel({
        chatMessageFormat: 'typesafe',
        inputs: { state: 'hello' },
        outputs: { result: 'custom response' },
      }),
    ).toBeNull();
  });
});
