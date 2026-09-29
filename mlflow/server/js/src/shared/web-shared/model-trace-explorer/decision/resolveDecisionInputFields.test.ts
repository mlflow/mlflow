import { describe, expect, it } from '@jest/globals';

import type { DecisionInputField } from './Decision.types';
import { resolveDecisionInputFields } from './resolveDecisionInputFields';

describe('resolveDecisionInputFields', () => {
  const fields = [
    { key: 'state', value: 'state value' },
    { key: 'questions', value: 'question value' },
    { key: 'model', value: 'model value' },
  ];

  it('resolves typed fields in view-model order and omits undeclared inputs', () => {
    const viewModels: DecisionInputField[] = [
      { kind: 'field', fieldKey: 'state' },
      { kind: 'questions', fieldKey: 'questions', items: [] },
      { kind: 'field', fieldKey: 'missing' },
    ];

    expect(resolveDecisionInputFields(fields, viewModels)).toEqual([
      { field: fields[0], viewModel: viewModels[0] },
      { field: fields[1], viewModel: viewModels[1] },
    ]);
  });

  it('falls back to every input when none of the declared fields exist', () => {
    expect(resolveDecisionInputFields(fields, [{ kind: 'field', fieldKey: 'missing' }])).toEqual(
      fields.map((field) => ({
        field,
        viewModel: { kind: 'field', fieldKey: field.key },
      })),
    );
  });
});
