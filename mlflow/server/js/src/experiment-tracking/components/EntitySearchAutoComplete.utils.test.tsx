import { describe, expect, test } from '@jest/globals';
import { cleanEntitySearchTagNames, getFilteredOptionsFromEntityName } from './EntitySearchAutoComplete.utils';
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

  describe('cleanEntitySearchTagNames', () => {
    test('hides internal mlflow.* tags by default', () => {
      expect(
        cleanEntitySearchTagNames(['mlflow.runName', 'mlflow.user', 'mlflow.note.content', 'my_user_tag']),
      ).toEqual(['my_user_tag']);
    });

    test('exposes git source tags as searchable internal tags', () => {
      expect(
        cleanEntitySearchTagNames([
          'mlflow.source.git.commit',
          'mlflow.source.git.branch',
          'mlflow.source.git.repoURL',
          'mlflow.runName',
        ]),
      ).toEqual(['`mlflow.source.git.commit`', '`mlflow.source.git.branch`', '`mlflow.source.git.repoURL`']);
    });
  });

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
