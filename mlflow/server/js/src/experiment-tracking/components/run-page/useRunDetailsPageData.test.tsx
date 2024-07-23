import { renderHook, act } from '@testing-library/react-for-react-18';
import { useRunDetailsPageData } from './useRunDetailsPageData';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';

import { getExperimentApi, getRunApi } from '../../actions';
import { searchModelVersionsApi } from '../../../model-registry/actions';
import { merge } from 'lodash';
import { ReduxState } from '../../../redux-types';
import { DeepPartial } from 'redux';

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

describe('useRunDetailsPageData', () => {
  const mountHook = (entities: DeepPartial<ReduxState['entities']> = {}, apis: DeepPartial<ReduxState['apis']> = {}) =>
    renderHook(() => useRunDetailsPageData(testRunUuid, testExperimentId), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <MockedReduxStoreProvider
          state={{
            entities: merge(
              {
                runInfosByUuid: {},
                experimentsById: {},
                tagsByRunUuid: {},
                latestMetricsByRunUuid: {},
                modelVersionsByRunUuid: {},
                paramsByRunUuid: {},
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
  });

  test("Start fetching run and experiment if it's not in the store already", () => {
    const { result } = mountHook();

    const { data } = result.current;

    expect(data.experiment).toBeUndefined();
    expect(data.runInfo).toBeUndefined();

    expect(getRunApi).toBeCalledWith(testRunUuid);
    expect(getExperimentApi).toBeCalledWith(testExperimentId);
    expect(searchModelVersionsApi).toBeCalledWith({ run_id: testRunUuid });
  });

  test("Do not fetch run if it's in the store already", () => {
    const { result } = mountHook({ runInfosByUuid: { [testRunUuid]: { runName: 'some_run' } } });

    const { data, loading } = result.current;

    expect(data.experiment).toBeUndefined();
    expect(data.runInfo).toEqual({ runName: 'some_run' });
    expect(loading).toEqual(true);

    expect(getRunApi).not.toBeCalled();
    expect(getExperimentApi).toBeCalledWith(testExperimentId);
  });

  test("Do not fetch experiment if it's in the store already", () => {
    const { result } = mountHook({
      experimentsById: { [testExperimentId]: { name: 'some_experiment' } },
    });

    const { data } = result.current;

    expect(data.runInfo).toBeUndefined();
    expect(data.experiment).toEqual({ name: 'some_experiment' });

    expect(getRunApi).toBeCalledWith(testRunUuid);
    expect(getExperimentApi).not.toBeCalled();
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

  test('To refetch the data when necessary', async () => {
    const { result } = mountHook();

    expect(getRunApi).toBeCalledTimes(1);

    await act(async () => {
      result.current.refetchRun();
    });

    expect(getRunApi).toBeCalledTimes(2);
  });
});
