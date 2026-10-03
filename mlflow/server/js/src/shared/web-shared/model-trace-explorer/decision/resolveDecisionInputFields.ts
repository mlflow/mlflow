import type { DecisionInputField } from './Decision.types';

interface TraceInputField {
  key: string;
}

export interface ResolvedDecisionInputField<T extends TraceInputField> {
  field: T;
  viewModel: DecisionInputField;
}

export const resolveDecisionInputFields = <T extends TraceInputField>(
  fields: readonly T[],
  viewModels: readonly DecisionInputField[],
): ResolvedDecisionInputField<T>[] => {
  const resolvedFields = viewModels.flatMap((viewModel) => {
    const field = fields.find(({ key }) => key === viewModel.fieldKey);
    return field ? [{ field, viewModel }] : [];
  });

  if (resolvedFields.length > 0) {
    return resolvedFields;
  }

  return fields.map((field) => ({
    field,
    viewModel: { kind: 'field', fieldKey: field.key },
  }));
};
