import { renderHook } from '../../../../common/utils/TestUtils.react18';
import { Provider, useSelector } from 'react-redux';
import type { DeepPartial } from 'redux';
import { createStore } from 'redux';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../fixtures/experiment-runs.fixtures';
import type { ExperimentRunsSelectorParams } from './experimentRuns.selector';
import { experimentRunsSelector } from './experimentRuns.selector';
import { LIFECYCLE_FILTER, MODEL_VERSION_FILTER } from '../../../types';

import type { ReduxState } from '../../../../redux-types';

describe('useExperimentRuns', () => {
  const mountComponentWithExperimentRuns = (
    experimentIds: string[],
    filterParams: DeepPartial<ExperimentRunsSelectorParams> = {},
  ) => {
    return renderHook(
      () =>
        useSelector((state: ReduxState) =>
          experimentRunsSelector(state, {
            experiments: experimentIds.map((id) => ({ experimentId: id })) as any,
            ...filterParams,
          }),
        ),
      {
        wrapper: ({ children }) => (
          <Provider store={createStore((s) => s as any, EXPERIMENT_RUNS_MOCK_STORE)}>{children}</Provider>
        ),
      },
    );
  };
  it('fetches single experiment runs from the store properly', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789']);

    expect(Object.keys(result.runInfos).length).toEqual(4);

    expect(Object.values(result.runInfos).map((r) => r.experimentId)).toEqual(expect.arrayContaining(['123456789']));
  });
  it('fetches experiment tags from the store properly', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789']);
    expect(result.experimentTags).toEqual(
      expect.objectContaining({
        'mlflow.experimentType': expect.objectContaining({
          key: 'mlflow.experimentType',
          value: 'NOTEBOOK',
        }),
        'mlflow.ownerEmail': expect.objectContaining({
          key: 'mlflow.ownerEmail',
          value: 'john.doe@databricks.com',
        }),
      }),
    );
  });
  it('fetches experiment runs tags from the store properly', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789']);

    expect(result.tagsList[0]).toEqual(
      expect.objectContaining({
        testtag1: expect.objectContaining({
          key: 'testtag1',
          value: 'value1',
        }),
      }),
    );

    expect(result.tagsList[1]).toEqual(
      expect.objectContaining({
        testtag2: expect.objectContaining({
          key: 'testtag2',
          value: 'value2_2',
        }),
      }),
    );
  });
  it('fetches metric and param keys list from the store properly', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789']);

    expect(result.metricKeyList).toEqual(['met1', 'met2', 'met3']);
    expect(result.paramKeyList).toEqual(['p1', 'p2', 'p3']);
  });
  it('fetches metrics list from the store properly', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789']);
    expect(result.metricsList[0]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'met1',
          value: 255,
          timestamp: 1000,
          step: 0,
        }),
      ]),
    );

    expect(result.metricsList[2]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        }),
      ]),
    );
  });
  it('fetches params list from the store properly', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789']);
    expect(result.paramsList[0]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'p1',
          value: '12',
        }),
      ]),
    );

    expect(result.paramsList[2]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'p2',
          value: '15',
        }),
      ]),
    );
  });

  it('fetches metrics for experiment without params', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['654321']);

    expect(result.metricKeyList).toEqual(['met1']);
    expect(result.paramKeyList).toEqual([]);

    expect(result.paramsList).toEqual([[]]);
    expect(result.metricsList[0]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        }),
      ]),
    );
  });

  it('fetches datasets for experiment runs', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789']);

    expect(result.datasetsList[0]).toEqual(
      EXPERIMENT_RUNS_MOCK_STORE.entities.runDatasetsByUuid['experiment123456789_run1'],
    );
  });

  it('filters runs with assigned model', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789'], {
      modelVersionFilter: MODEL_VERSION_FILTER.WITH_MODEL_VERSIONS,
    });

    expect(Object.keys(result.runInfos).length).toEqual(1);
  });

  it('filters runs without assigned model', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789'], {
      modelVersionFilter: MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS,
    });

    expect(Object.keys(result.runInfos).length).toEqual(3);
  });

  it('filters runs without datasets in datasetsFilter', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789'], {
      datasetsFilter: [{ experiment_id: '123456789', name: 'dataset_train', digest: 'abc' }],
    });

    expect(result.datasetsList.length).toEqual(1);
    expect(result.datasetsList[0]).toEqual(
      EXPERIMENT_RUNS_MOCK_STORE.entities.runDatasetsByUuid['experiment123456789_run1'],
    );
  });

  it('fetches only active runs by default', () => {
    const {
      result: { current: resultDefault },
    } = mountComponentWithExperimentRuns(['123456789']);

    const {
      result: { current: resultActive },
    } = mountComponentWithExperimentRuns(['123456789'], {
      lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
    });

    const {
      result: { current: resultDeleted },
    } = mountComponentWithExperimentRuns(['123456789'], {
      lifecycleFilter: LIFECYCLE_FILTER.DELETED,
    });

    expect(resultDefault.runInfos).toEqual(resultActive.runInfos);
    expect(resultDefault.runInfos).not.toEqual(resultDeleted.runInfos);
  });

  it('filters deleted runs', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['123456789'], {
      lifecycleFilter: LIFECYCLE_FILTER.DELETED,
    });

    expect(Object.keys(result.runInfos).length).toEqual(1);
  });

  it('fetches empty values for experiment with no runs and tags', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['789']);

    expect(result.experimentTags).toEqual({});
    expect(result.tagsList).toEqual([]);
    expect(result.runInfos).toEqual([]);
    expect(result.metricKeyList).toEqual([]);
    expect(result.paramKeyList).toEqual([]);
    expect(result.paramsList).toEqual([]);
    expect(result.metricsList).toEqual([]);
  });

  it('fetches empty values for not found experiment', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['55555']);

    expect(result.experimentTags).toEqual({});
    expect(result.tagsList).toEqual([]);
    expect(result.runInfos).toEqual([]);
    expect(result.metricKeyList).toEqual([]);
    expect(result.paramKeyList).toEqual([]);
    expect(result.paramsList).toEqual([]);
    expect(result.metricsList).toEqual([]);
  });

  it('fetches metrics, params, and tags with non-empty key and empty value, but not those with empty key', () => {
    const {
      result: { current: result },
    } = mountComponentWithExperimentRuns(['3210']);

    expect(result.metricsList.length).toEqual(1);
    expect(result.metricsList[0]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'met1',
          value: 2,
        }),
      ]),
    );

    expect(result.tagsList.length).toEqual(1);
    expect(result.tagsList[0]).toEqual(
      expect.objectContaining({
        testtag1: expect.objectContaining({
          key: 'testtag1',
          value: '',
        }),
      }),
    );

    expect(result.paramsList.length).toEqual(1);
    expect(result.paramsList[0]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'p1',
          value: '',
        }),
      ]),
    );
  });
});
