import { mount } from 'enzyme';
import { ATTRIBUTE_COLUMN_LABELS, COLUMN_TYPES } from '../../../constants';
import { SearchExperimentRunsFacetsState } from '../models/SearchExperimentRunsFacetsState';
import { useRunsColumnDefinitions, UseRunsColumnDefinitionsParams } from './experimentPage.column-utils';
import {
  EXPERIMENT_FIELD_PREFIX_METRIC,
  EXPERIMENT_FIELD_PREFIX_PARAM,
  EXPERIMENT_FIELD_PREFIX_TAG,
  makeCanonicalSortKey,
} from './experimentPage.common-utils';
import { ColDef, ColGroupDef } from '@ag-grid-community/core';

const getHookResult = (params: UseRunsColumnDefinitionsParams) => {
  let result = null;
  const Component = () => {
    result = useRunsColumnDefinitions(params);
    return null;
  };
  mount(<Component />);
  return result;
};

describe('ExperimentViewRuns column utils', () => {
  const MOCK_HOOK_PARAMS: UseRunsColumnDefinitionsParams = {} as any;

  const MOCK_METRICS = ['metric_1', 'metric_2'];
  const MOCK_PARAMS = ['param_1', 'param_2'];
  const MOCK_TAGS = ['tag_1', 'tag_2'];

  beforeEach(() => {
    Object.assign(MOCK_HOOK_PARAMS, {
      columnApi: { setColumnVisible: jest.fn() },
      compareExperiments: false,
      metricKeyList: MOCK_METRICS,
      paramKeyList: MOCK_PARAMS,
      tagKeyList: MOCK_TAGS,
      onExpand: jest.fn(),
      onSortBy: jest.fn(),
      onTogglePin: jest.fn(),
      selectedColumns: new SearchExperimentRunsFacetsState().selectedColumns,
    });
  });

  test('it creates proper column definitions with basic attributes', () => {
    const columnDefinitions = getHookResult(MOCK_HOOK_PARAMS);

    // Assert existence of regular attribute columns
    expect(columnDefinitions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.DATE,
          cellRenderer: 'DateCellRenderer',
        }),
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.DURATION,
        }),
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
          cellRenderer: 'RunNameCellRenderer',
        }),
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.USER,
        }),
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.VERSION,
          cellRenderer: 'VersionCellRenderer',
        }),
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.SOURCE,
          cellRenderer: 'SourceCellRenderer',
        }),
      ]),
    );

    // Assert existence of metric, param and tag columns
    expect(columnDefinitions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          children: expect.arrayContaining(
            MOCK_METRICS.map((key) =>
              expect.objectContaining({
                colId: makeCanonicalSortKey(COLUMN_TYPES.METRICS, key),
                field: `${EXPERIMENT_FIELD_PREFIX_METRIC}-${key}`,
              }),
            ),
          ),
        }),
        expect.objectContaining({
          children: expect.arrayContaining(
            MOCK_PARAMS.map((key) =>
              expect.objectContaining({
                colId: makeCanonicalSortKey(COLUMN_TYPES.PARAMS, key),
                field: `${EXPERIMENT_FIELD_PREFIX_PARAM}-${key}`,
              }),
            ),
          ),
        }),
        expect.objectContaining({
          children: expect.arrayContaining(
            MOCK_TAGS.map((key) =>
              expect.objectContaining({
                colId: makeCanonicalSortKey(COLUMN_TYPES.TAGS, key),
                field: `${EXPERIMENT_FIELD_PREFIX_TAG}-${key}`,
              }),
            ),
          ),
        }),
      ]),
    );

    // We're not comparing experiments so experiment name should be hidden
    expect(columnDefinitions).toEqual(
      expect.not.arrayContaining([
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.EXPERIMENT_NAME,
        }),
      ]),
    );
  });

  test('it displays experiment name column when necessary', () => {
    const columnDefinitions = getHookResult({ ...MOCK_HOOK_PARAMS, compareExperiments: true });

    // When comparing experiments, we should display experiment name column as well
    expect(columnDefinitions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          headerName: ATTRIBUTE_COLUMN_LABELS.EXPERIMENT_NAME,
        }),
      ]),
    );
  });

  test('it hides and shows certain metrics and param columns', () => {
    const hookParams = MOCK_HOOK_PARAMS;
    const Component = (props: { hookParams: UseRunsColumnDefinitionsParams }) => {
      useRunsColumnDefinitions(props.hookParams);
      return null;
    };
    // Initialize and mount the component with initial set of params
    const wrapper = mount(<Component hookParams={hookParams} />);

    // Next, select some columns by changing the props of the component
    hookParams.selectedColumns = ['metrics.`metric_1`', 'params.`param_2`'];
    wrapper.setProps({
      hookParams,
    });

    const setColumnVisibleMock = MOCK_HOOK_PARAMS.columnApi?.setColumnVisible;

    // Assert that setColumnVisible() has been called with "true" for metric_1 and param_2...
    expect(setColumnVisibleMock).toBeCalledWith(makeCanonicalSortKey(COLUMN_TYPES.METRICS, 'metric_1'), true);
    expect(setColumnVisibleMock).toBeCalledWith(makeCanonicalSortKey(COLUMN_TYPES.PARAMS, 'param_2'), true);

    // ...but has not for the remaining columns
    expect(setColumnVisibleMock).not.toBeCalledWith(makeCanonicalSortKey(COLUMN_TYPES.METRICS, 'metric_2'), true);

    expect(setColumnVisibleMock).not.toBeCalledWith(makeCanonicalSortKey(COLUMN_TYPES.PARAMS, 'param_1'), true);
  });

  test('remembers metric/param/tag keys even if they are not in the newly fetched set', () => {
    // Let's start with initializing the component with only one known metric key: "metric_1"
    const hookParams: UseRunsColumnDefinitionsParams = {
      ...MOCK_HOOK_PARAMS,
      metricKeyList: ['metric_1'],
    };
    let result: ColGroupDef[] = [];
    const Component = (props: { hookParams: UseRunsColumnDefinitionsParams }) => {
      result = useRunsColumnDefinitions(props.hookParams) as ColGroupDef[];
      return null;
    };
    const wrapper = mount(<Component hookParams={hookParams} />);

    // Assert single metric column in the result set
    expect(result.find((r) => r.groupId === COLUMN_TYPES.METRICS)?.children?.map(({ colId }: ColDef) => colId)).toEqual(
      ['metrics.`metric_1`'],
    );

    // Next, add a new set of two metrics
    wrapper.setProps({
      hookParams: { ...hookParams, metricKeyList: ['metric_1', 'metric_2'] },
    });

    // Assert two metric columns in the result set
    expect(result.find((r) => r.groupId === COLUMN_TYPES.METRICS)?.children?.map(({ colId }: ColDef) => colId)).toEqual(
      ['metrics.`metric_1`', 'metrics.`metric_2`'],
    );

    // Finally, retract the first metric and leavy "metric_2" only
    wrapper.setProps({
      hookParams: { ...hookParams, metricKeyList: ['metric_2'] },
    });

    // We expect previous metric column to still exist - this ensures that columns won't
    // disappear on the new dataset without certain metric/param/tag keys
    expect(result.find((r) => r.groupId === COLUMN_TYPES.METRICS)?.children?.map(({ colId }: ColDef) => colId)).toEqual(
      ['metrics.`metric_1`', 'metrics.`metric_2`'],
    );
  });
});
