import { renderHook } from '@testing-library/react';
import { useRunDetailsPageData } from './useRunDetailsPageData';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';

import { merge } from 'lodash';
import { ReduxState } from '../../../../redux-types';
import { DeepPartial } from 'redux';

const mockAction = (id: string) => ({ type: 'action', payload: Promise.resolve(), meta: { id } });

jest.mock('../../../actions', () => ({
  getExperimentApi: jest.fn(() => mockAction('experiment_request')),
  getRunApi: jest.fn(() => mockAction('run_request')),
}));

jest.mock('../../../../model-registry/actions', () => ({
  searchModelVersionsApi: jest.fn(() => mockAction('models_request')),
}));

const testRunUuid = 'test-run-uuid';
const testExperimentId = '12345';

describe('useRunDetailsPageData', () => {
  const mountHook = (entities: DeepPartial<ReduxState['entities']> = {}, apis: DeepPartial<ReduxState['apis']> = {}) =>
    renderHook(() => useRunDetailsPageData({ runUuid: testRunUuid, experimentId: testExperimentId }), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <MockedReduxStoreProvider
          state={{
            entities: merge(
              {
                runInfosByUuid: {},
                experimentsById: {},
                tagsByRunUuid: {
                  [testRunUuid]: [
                    { key: 'testtag1', value: '' },
                    { key: '\t', value: 'value1' },
                  ],
                },
                latestMetricsByRunUuid: {
                  [testRunUuid]: [
                    { key: 'met1', value: 2, timestamp: 1000, step: 0 },
                    { key: '', value: 0, timestamp: 1000, step: 0 },
                  ],
                },
                modelVersionsByRunUuid: {},
                paramsByRunUuid: {
                  [testRunUuid]: [
                    { key: 'p1', value: '' },
                    { key: '\n', value: '0' },
                  ],
                },
                runDatasetsByUuid: {
                  [testRunUuid]: [
                    {
                      dataset: {
                        digest: 'digest',
                        name: 'name',
                        profile: 'profile',
                        schema: 'schema',
                        source: 'source',
                        sourceType: 'sourceType',
                      },
                      tags: [{ key: 'tag1', value: 'value1' }],
                    },
                  ],
                },
              },
              entities,
            ),
            apis,
          }}
        >
          {children}
        </MockedReduxStoreProvider>
      ),
    });

  test('Fetches metrics, params, and tags with non-empty key and empty value, but not those with empty key', () => {
    const { result } = mountHook();
    const { tags, params, latestMetrics, datasets } = result.current;

    expect(tags).toEqual({ '0': { key: 'testtag1', value: '' } });
    expect(params).toEqual({ '0': { key: 'p1', value: '' } });
    expect(latestMetrics).toEqual({ '0': { key: 'met1', value: 2, timestamp: 1000, step: 0 } });
    expect(datasets).toEqual([
      {
        dataset: {
          digest: 'digest',
          name: 'name',
          profile: 'profile',
          schema: 'schema',
          source: 'source',
          sourceType: 'sourceType',
        },
        tags: [{ key: 'tag1', value: 'value1' }],
      },
    ]);
  });
});
