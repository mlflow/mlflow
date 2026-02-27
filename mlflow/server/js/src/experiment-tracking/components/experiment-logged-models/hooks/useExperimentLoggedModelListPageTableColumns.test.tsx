import { describe, it, expect, jest } from '@jest/globals';
import { renderHook, render, screen, waitFor } from '@testing-library/react';
import type { LoggedModelProto, LoggedModelMetricProto } from '../../../types';
import {
  useExperimentLoggedModelListPageTableColumns,
  parseLoggedModelMetricOrderByColumnId,
} from './useExperimentLoggedModelListPageTableColumns';
import { IntlProvider } from 'react-intl';
import type { ColDef, ColGroupDef } from '@ag-grid-community/core';
import React from 'react';
import { ExperimentLoggedModelOpenDatasetDetailsContext } from './useExperimentLoggedModelOpenDatasetDetails';
import userEvent from '@testing-library/user-event';

const createTestMetric = (
  key: string,
  value: number,
  dataset_name?: string,
  index?: number,
): LoggedModelMetricProto => ({
  key,
  value,
  dataset_digest: '123',
  dataset_name,
  run_id: (123 + (index ?? 0)).toString(),
  step: 1,
  timestamp: 1728322600000,
});

describe('useExperimentLoggedModelListPageTableColumns', () => {
  describe('grouping metrics by datasets', () => {
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
          createTestMetric('metric-1', (index + 1) * 100 + 10 + 1, 'dataset-1', index),
          createTestMetric('metric-1', (index + 1) * 100 + 10 + 2, 'dataset-2', index),
          createTestMetric('metric-1', (index + 1) * 100 + 10 + 3, 'dataset-3', index),

          // Metric 2 will be logged only in dataset-1, 100s will be be model index, 10s will be metric name and 1s will be dataset index
          createTestMetric('metric-2', (index + 1) * 100 + 20 + 1, 'dataset-2', index),

          // Metric 3 will have no dataset set
          createTestMetric('metric-3', (index + 1) * 100 + 20 + 1, undefined, index),
        ],
        params: undefined,
      },
    }));

    const renderTestHook = () => {
      const { columnDefs } = renderHook(() => useExperimentLoggedModelListPageTableColumns({ loggedModels }), {
        wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
      }).result.current;

      return columnDefs;
    };

    it('should group metrics by dataset name', () => {
      const hookResult = renderTestHook();

      const datasetNames = ['No dataset', 'dataset-1 (#123)', 'dataset-2 (#123)', 'dataset-3 (#123)'];

      // Let's get the column groups that are related to metrics
      const metricColumnGroups = hookResult.filter(
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
          'dataset-1 (#123)': {
            'metric-1': 111,
            'metric-2': undefined,
          },
          'dataset-2 (#123)': {
            'metric-1': 112,
            'metric-2': 121,
          },
          'dataset-3 (#123)': {
            'metric-1': 113,
            'metric-2': undefined,
          },
        },
        {
          // metric-3 should be ungrouped
          '': {
            'metric-3': 221,
          },
          'dataset-1 (#123)': {
            'metric-1': 211,
            'metric-2': undefined,
          },
          'dataset-2 (#123)': {
            'metric-1': 212,
            'metric-2': 221,
          },
          'dataset-3 (#123)': {
            'metric-1': 213,
            'metric-2': undefined,
          },
        },
        {
          // metric-3 should be ungrouped
          '': {
            'metric-3': 321,
          },
          'dataset-1 (#123)': {
            'metric-1': 311,
            'metric-2': undefined,
          },
          'dataset-2 (#123)': {
            'metric-1': 312,
            'metric-2': 321,
          },
          'dataset-3 (#123)': {
            'metric-1': 313,
            'metric-2': undefined,
          },
        },
      ]);
    });

    it('should render clickable column headers', async () => {
      const hookResult = renderTestHook();
      const onDatasetClicked = jest.fn();

      const datasetNames = ['No dataset', 'dataset-1 (#123)', 'dataset-2 (#123)', 'dataset-3 (#123)'];

      // Let's get the column groups that are related to metrics
      const metricColumnGroups = hookResult
        .filter(
          (columnDef) =>
            columnDef.headerName === '' || (columnDef.headerName && datasetNames.includes(columnDef.headerName)),
        )
        .map((colGroup) => ({
          ...colGroup,
          getGroupId: () => (colGroup as ColGroupDef).groupId,
        })) as ColGroupDef[];

      render(
        <div>
          {metricColumnGroups.map((columnGroup) => (
            <div key={columnGroup.groupId}>
              {React.createElement(columnGroup.headerGroupComponent, { columnGroup })}
            </div>
          ))}
        </div>,
        {
          wrapper: ({ children }) => (
            // @ts-expect-error Type 'unknown' is not assignable to type 'Promise<void>'
            <ExperimentLoggedModelOpenDatasetDetailsContext.Provider value={{ onDatasetClicked }}>
              <IntlProvider locale="en">{children}</IntlProvider>
            </ExperimentLoggedModelOpenDatasetDetailsContext.Provider>
          ),
        },
      );

      // Wait for the particular dataset header to be rendered
      await waitFor(() => {
        expect(screen.getByText('dataset-1 (#123)')).toBeVisible();
      });

      // Assert other dataset names are also visible
      expect(screen.getByText('No dataset')).toBeVisible();
      expect(screen.getByText('dataset-2 (#123)')).toBeVisible();
      expect(screen.getByText('dataset-3 (#123)')).toBeVisible();

      // Click on the dataset-1 header
      await userEvent.click(screen.getByText('dataset-1 (#123)'));

      // Assert the dataset click callback was called with the dataset name and digest
      // and with the first found run id
      await waitFor(() => {
        expect(onDatasetClicked).toHaveBeenCalledWith({
          datasetName: 'dataset-1',
          datasetDigest: '123',
          runId: '123',
        });
      });
    });
  });

  describe('parseLoggedModelMetricOrderByColumnId', () => {
    describe('with dataset names containing dots', () => {
      it('should correctly parse dataset names with single dot', () => {
        const columnId = 'metrics.{"metricKey":"accuracy","datasetName":"dataset.v1","datasetDigest":"abc123"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: 'accuracy',
          datasetName: 'dataset.v1',
          datasetDigest: 'abc123',
        });
      });

      it('should correctly parse dataset names with multiple dots', () => {
        const columnId =
          'metrics.{"metricKey":"f1_score","datasetName":"my.dataset.v2.final","datasetDigest":"xyz789"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: 'f1_score',
          datasetName: 'my.dataset.v2.final',
          datasetDigest: 'xyz789',
        });
      });

      it('should correctly parse dataset names with dots at start and end', () => {
        const columnId = 'metrics.{"metricKey":"precision","datasetName":".dataset.name.","datasetDigest":"digest456"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: 'precision',
          datasetName: '.dataset.name.',
          datasetDigest: 'digest456',
        });
      });

      it('should correctly parse dataset names with consecutive dots', () => {
        const columnId = 'metrics.{"metricKey":"recall","datasetName":"dataset..name","datasetDigest":"hash999"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: 'recall',
          datasetName: 'dataset..name',
          datasetDigest: 'hash999',
        });
      });
    });

    describe('with metric keys containing dots', () => {
      it('should correctly parse metric keys with dots', () => {
        const columnId =
          'metrics.{"metricKey":"metrics.accuracy.train","datasetName":"training_set","datasetDigest":"abc123"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: 'metrics.accuracy.train',
          datasetName: 'training_set',
          datasetDigest: 'abc123',
        });
      });

      it('should correctly parse metric keys with dots and no dataset', () => {
        const columnId = 'metrics.{"metricKey":"eval.metrics.f1"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: 'eval.metrics.f1',
          datasetName: undefined,
          datasetDigest: undefined,
        });
      });
    });

    describe('with ungrouped metrics', () => {
      it('should correctly parse ungrouped metrics without dataset', () => {
        const columnId = 'metrics.{"metricKey":"loss"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: 'loss',
          datasetName: undefined,
          datasetDigest: undefined,
        });
      });
    });

    describe('with non-metric column IDs', () => {
      it('should return fallback for column ID without prefix', () => {
        const columnId = 'created_time';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: columnId,
          datasetName: undefined,
          datasetDigest: undefined,
        });
      });

      it('should return fallback for invalid JSON', () => {
        const columnId = 'metrics.not-valid-json';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: columnId,
          datasetName: undefined,
          datasetDigest: undefined,
        });
      });

      it('should return fallback for JSON missing metricKey', () => {
        const columnId = 'metrics.{"datasetName":"dataset","datasetDigest":"digest"}';

        const result = parseLoggedModelMetricOrderByColumnId(columnId);

        expect(result).toEqual({
          metricKey: columnId,
          datasetName: undefined,
          datasetDigest: undefined,
        });
      });
    });
  });
});
