import { describe, expect, test } from '@jest/globals';
import {
  getEntitySearchOptionsFromEntityNames,
  getFilteredOptionsFromEntityName,
} from './EntitySearchAutoComplete.utils';
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

  test('quotes dynamic entity names in autocomplete values', () => {
    const options = getEntitySearchOptionsFromEntityNames(
      {
        metricNames: ['account_match/mean', 'f1 score'],
        paramNames: ['model.version'],
        tagNames: ['mlflow.user'],
      },
      [],
    );

    expect(options).toEqual([
      {
        label: 'Metrics',
        options: [
          { label: 'metrics.account_match/mean', value: 'metrics.`account_match/mean`' },
          { label: 'metrics.f1 score', value: 'metrics.`f1 score`' },
        ],
      },
      { label: 'Parameters', options: [{ label: 'params.model.version', value: 'params.`model.version`' }] },
      { label: 'Tags', options: [{ label: 'tags.mlflow.user', value: 'tags.`mlflow.user`' }] },
      { label: 'Attributes', options: [] },
    ]);

    expect(
      getFilteredOptionsFromEntityName(
        options,
        { name: 'metrics.account_match', startIndex: 0, endIndex: 21 },
        { Metrics: 10, Parameters: 10, Tags: 10, Attributes: 10 },
      )[0].options[0].value,
    ).toBe('metrics.`account_match/mean`');
  });

  test('matches the quoted prefix of an inserted identifier', () => {
    const options = getEntitySearchOptionsFromEntityNames(
      { metricNames: [], paramNames: [], tagNames: ['team.owner'] },
      [],
    );

    expect(
      getFilteredOptionsFromEntityName(
        options,
        { name: 'tags.`team.own', startIndex: 0, endIndex: 14 },
        { Metrics: 10, Parameters: 10, Tags: 10, Attributes: 10 },
      ),
    ).toEqual([
      {
        label: 'Tags',
        options: [expect.objectContaining({ value: 'tags.`team.owner`' })],
      },
    ]);
  });
});
