import { SearchUtils } from './SearchUtils';

test("validateSearchInput", () => {
  const goodComparators = {
    'metrics': ['<', '>', '<=', '>=', '=', '!='],
    'params': ['=', '!='],
    'tags': ['=', '!=']
  };
  const goodValues = {
    'metrics': '0.123456789',
    'params': '"1.0"',
    'tags': '"abc"',
  };
  const goodKeys = {
    'metrics': 'rmse',
    'params': 'alpha',
    'tags': 'mlflow.parentRunId',
  };
  const badComparators = {
    'metrics': ['~', '~=', '=~'],
    'params': ['>', '<', '>=', '<='],
    'tags': ['>', '<', '>=', '<='],
  };
  ['metrics', 'params', 'tags'].forEach((entityType) => {
    const key = goodKeys[entityType];
    const value = goodValues[entityType];
    // Test searches with valid comparators for the current entity type
    goodComparators[entityType].forEach((goodComparator) => {
      SearchUtils.validateSearchInput(`${entityType}.${key} ${goodComparator} ${value}`);
    });
    // Test searches with invalid comparators for the current entity type
    badComparators[entityType].forEach((badComparator) => {
      expect(() => {
        SearchUtils.validateSearchInput(`${entityType}.${key} ${badComparator} ${value}`);
      }).toThrow('The search input should be like');
    });
  });
  // Test searching over a combination of entity types
  SearchUtils.validateSearchInput('metrics.rmse < 0.1 and tags.mlflow.parentRunId = "abc"');
  SearchUtils.validateSearchInput('metrics.rmse < 0.1 and ' +
    'tags.mlflow.parentRunId = "abc" and params.alpha = "0.1"');
  expect(() => {
    SearchUtils.validateSearchInput('metrics.rmse < 0.1 and tags.mlflow.parentRunId ~= "abc"');
  }).toThrow('The search input should be like');
});
