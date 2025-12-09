import { describe, expect, test } from '@jest/globals';
import { getFilteredOptionsFromEntityName } from './EntitySearchAutoComplete.utils';
import type {
  EntitySearchAutoCompleteEntity,
  EntitySearchAutoCompleteOptionGroup,
} from './EntitySearchAutoComplete.utils';

describe('EntitySearchAutoComplete utils', () => {
  const baseOptions: EntitySearchAutoCompleteOptionGroup[] = [
    {
      label: 'Metrics',
      options: [{ value: 'metrics.accuracy' }, { value: 'metrics.loss' }, { value: 'metrics.precision(micro)' }],
    },
  ];

  const suggestionLimits = {
    Metrics: 10,
  };

  test('getFilteredOptionsFromEntityName handles single open parenthesis without crashing', () => {
    const entityBeingEdited: EntitySearchAutoCompleteEntity = {
      name: 'precision(',
      startIndex: 0,
      endIndex: 10,
    };

    // Should not throw an error
    expect(() => {
      const result = getFilteredOptionsFromEntityName(baseOptions, entityBeingEdited, suggestionLimits);
      expect(result).toBeDefined();
    }).not.toThrow();
  });
});
