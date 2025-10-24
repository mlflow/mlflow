import { getEntitySearchOptionsFromEntityNames } from './EntitySearchAutoComplete.utils';

describe('EntitySearchAutoComplete.utils', () => {
  describe('getEntitySearchOptionsFromEntityNames', () => {
    const attributeOptions = [{ value: 'attributes.run_id' }, { value: 'attributes.run_name' }];

    test('it quotes metric names with spaces using backticks', () => {
      const entityNames = {
        metricNames: ['accuracy', 'avg model score', 'test metric'],
        paramNames: [],
        tagNames: [],
      };

      const result = getEntitySearchOptionsFromEntityNames(entityNames, attributeOptions);

      const metricsGroup = result.find((group) => group.label === 'Metrics');
      expect(metricsGroup?.options).toEqual([
        { value: 'metrics.accuracy' },
        { value: 'metrics.`avg model score`' },
        { value: 'metrics.`test metric`' },
      ]);
    });

    test('it quotes param names with spaces using backticks', () => {
      const entityNames = {
        metricNames: [],
        paramNames: ['learning_rate', 'model type', 'batch size'],
        tagNames: [],
      };

      const result = getEntitySearchOptionsFromEntityNames(entityNames, attributeOptions);

      const paramsGroup = result.find((group) => group.label === 'Parameters');
      expect(paramsGroup?.options).toEqual([
        { value: 'params.learning_rate' },
        { value: 'params.`model type`' },
        { value: 'params.`batch size`' },
      ]);
    });

    test('it quotes names with dots using backticks', () => {
      const entityNames = {
        metricNames: ['metric.with.dots'],
        paramNames: ['param.with.dots'],
        tagNames: [],
      };

      const result = getEntitySearchOptionsFromEntityNames(entityNames, attributeOptions);

      const metricsGroup = result.find((group) => group.label === 'Metrics');
      expect(metricsGroup?.options).toEqual([{ value: 'metrics.`metric.with.dots`' }]);

      const paramsGroup = result.find((group) => group.label === 'Parameters');
      expect(paramsGroup?.options).toEqual([{ value: 'params.`param.with.dots`' }]);
    });

    test('it quotes names with double quotes using backticks', () => {
      const entityNames = {
        metricNames: ['metric"with"quotes'],
        paramNames: ['param"with"quotes'],
        tagNames: [],
      };

      const result = getEntitySearchOptionsFromEntityNames(entityNames, attributeOptions);

      const metricsGroup = result.find((group) => group.label === 'Metrics');
      expect(metricsGroup?.options).toEqual([{ value: 'metrics.`metric"with"quotes`' }]);

      const paramsGroup = result.find((group) => group.label === 'Parameters');
      expect(paramsGroup?.options).toEqual([{ value: 'params.`param"with"quotes`' }]);
    });

    test('it quotes names with backticks using double quotes', () => {
      const entityNames = {
        metricNames: ['metric`with`backticks'],
        paramNames: ['param`with`backticks'],
        tagNames: [],
      };

      const result = getEntitySearchOptionsFromEntityNames(entityNames, attributeOptions);

      const metricsGroup = result.find((group) => group.label === 'Metrics');
      expect(metricsGroup?.options).toEqual([{ value: 'metrics."metric`with`backticks"' }]);

      const paramsGroup = result.find((group) => group.label === 'Parameters');
      expect(paramsGroup?.options).toEqual([{ value: 'params."param`with`backticks"' }]);
    });

    test('it does not quote simple names', () => {
      const entityNames = {
        metricNames: ['accuracy', 'loss'],
        paramNames: ['learning_rate', 'epochs'],
        tagNames: [],
      };

      const result = getEntitySearchOptionsFromEntityNames(entityNames, attributeOptions);

      const metricsGroup = result.find((group) => group.label === 'Metrics');
      expect(metricsGroup?.options).toEqual([{ value: 'metrics.accuracy' }, { value: 'metrics.loss' }]);

      const paramsGroup = result.find((group) => group.label === 'Parameters');
      expect(paramsGroup?.options).toEqual([{ value: 'params.learning_rate' }, { value: 'params.epochs' }]);
    });
  });
});
