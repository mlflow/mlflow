import { renderHook } from '@testing-library/react';
import { LoggedModelProto, LoggedModelMetricProto } from '../../../types';
import { useExperimentLoggedModelListPageTableColumns } from './useExperimentLoggedModelListPageTableColumns';
import { IntlProvider } from 'react-intl';
import { ColDef, ColGroupDef } from '@ag-grid-community/core';

const createTestMetric = (key: string, value: number, dataset_name?: string): LoggedModelMetricProto => ({
  key,
  value,
  dataset_digest: '123',
  dataset_name,
  run_id: '123',
  step: 1,
  timestamp: 1728322600000,
});

describe('useExperimentLoggedModelListPageTableColumns', () => {
  describe('grouping metrics by datasets', () => {
    it('should group metrics by dataset name', () => {
      const loggedModels: LoggedModelProto[] = new Array(3).fill(0).map<LoggedModelProto>((_, index) => ({
        __typename: 'MlflowLoggedModel',
        info: {
          modelId: 'test-model-id',
        } as any,
        data: {
          __typename: 'MlflowLoggedModelData',
          // Let's prepare some easibly testable metrics
          metrics: [
            // Metric 1 will be logged in all 3 datasets, 100s will be be model index, 10s will be metric name and 1s will be dataset index
            createTestMetric('metric-1', (index + 1) * 100 + 10 + 1, 'dataset-1'),
            createTestMetric('metric-1', (index + 1) * 100 + 10 + 2, 'dataset-2'),
            createTestMetric('metric-1', (index + 1) * 100 + 10 + 3, 'dataset-3'),

            // Metric 2 will be logged only in dataset-1, 100s will be be model index, 10s will be metric name and 1s will be dataset index
            createTestMetric('metric-2', (index + 1) * 100 + 20 + 1, 'dataset-2'),

            // Metric 3 will have no dataset set
            createTestMetric('metric-3', (index + 1) * 100 + 20 + 1, undefined),
          ],
          params: undefined,
        },
      }));

      const datasetNames = ['dataset-1', 'dataset-2', 'dataset-3'];

      const columnGroupDefs = renderHook(() => useExperimentLoggedModelListPageTableColumns({ loggedModels }), {
        wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
      }).result.current;

      // Let's get the column groups that are related to metrics
      const metricColumnGroups = columnGroupDefs.filter(
        (columnDef) =>
          columnDef.headerName === '' || (columnDef.headerName && datasetNames.includes(columnDef.headerName)),
      );

      const renderCell = (column: ColDef | ColGroupDef, data: LoggedModelProto) =>
        'valueGetter' in column && typeof column.valueGetter === 'function' && column.valueGetter({ data } as any);

      // Render our table rows
      const rows = loggedModels.map((loggedModel) =>
        metricColumnGroups.reduce(
          (result, columnGroup) => ({
            ...result,
            [columnGroup.headerName ?? '']: ('children' in columnGroup
              ? (columnGroup.children as ColDef[])
              : []
            ).reduce(
              (groupResult, metricColumn) => ({
                ...groupResult,
                [metricColumn.headerName ?? '']: renderCell(metricColumn, loggedModel),
              }),
              {},
            ),
          }),
          {},
        ),
      );

      expect(rows).toEqual([
        {
          // metric-3 should be ungrouped
          '': {
            'metric-3': 121,
          },
          'dataset-1': {
            'metric-1': 111,
            'metric-2': undefined,
          },
          'dataset-2': {
            'metric-1': 112,
            'metric-2': 121,
          },
          'dataset-3': {
            'metric-1': 113,
            'metric-2': undefined,
          },
        },
        {
          // metric-3 should be ungrouped
          '': {
            'metric-3': 221,
          },
          'dataset-1': {
            'metric-1': 211,
            'metric-2': undefined,
          },
          'dataset-2': {
            'metric-1': 212,
            'metric-2': 221,
          },
          'dataset-3': {
            'metric-1': 213,
            'metric-2': undefined,
          },
        },
        {
          // metric-3 should be ungrouped
          '': {
            'metric-3': 321,
          },
          'dataset-1': {
            'metric-1': 311,
            'metric-2': undefined,
          },
          'dataset-2': {
            'metric-1': 312,
            'metric-2': 321,
          },
          'dataset-3': {
            'metric-1': 313,
            'metric-2': undefined,
          },
        },
      ]);
    });
  });
});
