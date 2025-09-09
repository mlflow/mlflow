import { renderHook, act, waitFor } from '@testing-library/react';
import { useRunDetailsPageDataLegacy } from './useRunDetailsPageDataLegacy';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';

import { getExperimentApi, getRunApi } from '../../actions';
import { searchModelVersionsApi } from '../../../model-registry/actions';
import { merge } from 'lodash';
import type { ReduxState } from '../../../redux-types';
import type { DeepPartial } from 'redux';
import Utils from '../../../common/utils/Utils';

const mockAction = (id: string) => ({ type: 'action', payload: Promise.resolve(), meta: { id } });

jest.mock('../../actions', () => ({
  getExperimentApi: jest.fn(() => mockAction('experiment_request')),
  getRunApi: jest.fn(() => mockAction('run_request')),
}));

jest.mock('../../../model-registry/actions', () => ({
  searchModelVersionsApi: jest.fn(() => mockAction('models_request')),
}));

const testRunUuid = 'test-run-uuid';
const testExperimentId = '12345';

describe('useRunDetailsPageDataLegacy', () => {
  const mountHook = (entities: DeepPartial<ReduxState['entities']> = {}, apis: DeepPartial<ReduxState['apis']> = {}) =>
    renderHook(() => useRunDetailsPageDataLegacy(testRunUuid, testExperimentId), {
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

  beforeEach(() => {
    jest.mocked(getRunApi).mockClear();
    jest.mocked(getExperimentApi).mockClear();
    jest.mocked(searchModelVersionsApi).mockClear();

    jest.mocked(getRunApi).mockImplementation(() => mockAction('run_request') as any);
    jest.mocked(getExperimentApi).mockImplementation(() => mockAction('experiment_request') as any);
    jest.spyOn(Utils, 'logErrorAndNotifyUser');
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test("Start fetching run and experiment if it's not in the store already", () => {
    const { result } = mountHook();

    const { data } = result.current;

    expect(data.experiment).toBeUndefined();
    expect(data.runInfo).toBeUndefined();

    expect(getRunApi).toHaveBeenCalledWith(testRunUuid);
    expect(getExperimentApi).toHaveBeenCalledWith(testExperimentId);
    expect(searchModelVersionsApi).toHaveBeenCalledWith({ run_id: testRunUuid });
  });

  test("Do not fetch run if it's in the store already", () => {
    const { result } = mountHook({ runInfosByUuid: { [testRunUuid]: { runName: 'some_run' } } });

    const { data, loading } = result.current;

    expect(data.experiment).toBeUndefined();
    expect(data.runInfo).toEqual({ runName: 'some_run' });
    expect(loading).toEqual(true);

    expect(getRunApi).not.toHaveBeenCalled();
    expect(getExperimentApi).toHaveBeenCalledWith(testExperimentId);
  });

  test("Do not fetch experiment if it's in the store already", () => {
    const { result } = mountHook({
      experimentsById: { [testExperimentId]: { name: 'some_experiment' } },
    });

    const { data } = result.current;

    expect(data.runInfo).toBeUndefined();
    expect(data.experiment).toEqual({ name: 'some_experiment' });

    expect(getRunApi).toHaveBeenCalledWith(testRunUuid);
    expect(getExperimentApi).not.toHaveBeenCalled();
  });

  test('Properly conveys get experiment API error if there is one', () => {
    const experimentFetchError = new Error('experiment_fetch_error');
    const { result } = mountHook({}, { experiment_request: { active: false, error: experimentFetchError } });

    expect(result.current.errors.experimentFetchError).toBe(experimentFetchError);
  });

  test('Properly conveys get run API error if there is one', () => {
    const runFetchError = new Error('run_fetch_error');
    const { result } = mountHook({}, { run_request: { active: false, error: runFetchError } });

    expect(result.current.errors.runFetchError).toBe(runFetchError);
  });

  test('Properly reports experiment fetch error if there is one', async () => {
    jest.mocked(getExperimentApi).mockImplementation(() => {
      return {
        type: 'get_experiment',
        meta: { id: 'abc' },
        payload: Promise.reject(new Error('experiment_fetch_error')),
      };
    });
    jest.spyOn(Utils, 'logErrorAndNotifyUser');

    mountHook();

    await waitFor(() => {
      expect(jest.mocked(Utils.logErrorAndNotifyUser).mock.lastCall?.[0]).toBeInstanceOf(Error);
      expect(jest.mocked(Utils.logErrorAndNotifyUser).mock.lastCall?.[0].message).toEqual('experiment_fetch_error');
    });
  });

  test('Properly reports run fetch error if there is one', async () => {
    jest.mocked(getRunApi).mockImplementation(() => {
      return {
        type: 'get_run',
        meta: { id: 'abc' },
        payload: Promise.reject(new Error('run_fetch_error')),
      };
    });
    mountHook();

    await waitFor(() => {
      expect(jest.mocked(Utils.logErrorAndNotifyUser).mock.lastCall?.[0]).toBeInstanceOf(Error);
      expect(jest.mocked(Utils.logErrorAndNotifyUser).mock.lastCall?.[0].message).toEqual('run_fetch_error');
    });
  });

  test('To refetch the data when necessary', async () => {
    const { result } = mountHook();

    expect(getRunApi).toHaveBeenCalledTimes(1);

    await act(async () => {
      result.current.refetchRun();
    });

    expect(getRunApi).toHaveBeenCalledTimes(2);
  });

  test('Fetches metrics, params, and tags with non-empty key and empty value, but not those with empty key', () => {
    const { result } = mountHook();
    const { tags, params, latestMetrics, datasets } = result.current.data;

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
