import { describe, expect, it } from '@jest/globals';

import { resolveDecisionViewModel } from './resolveDecisionViewModel';

describe('resolveDecisionViewModel', () => {
  it('returns null when no registered decision translator matches', () => {
    expect(resolveDecisionViewModel()).toBeNull();
    expect(
      resolveDecisionViewModel({
        title: 'openai.chat.completions',
        modelProvider: 'openai',
        inputs: { questions: { relevant: { type: 'noul' } } },
        outputs: { answers: { relevant: { type: 'noul', noul: 0.9 } } },
      }),
    ).toBeNull();
  });

  it('uses the matching translator to build a provider-neutral view model', () => {
    expect(
      resolveDecisionViewModel({
        title: 'typesafe.system_one',
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
        title: 'typesafe.system_one',
        inputs: { state: 'hello' },
        outputs: { result: 'custom response' },
      }),
    ).toBeNull();
  });
});
