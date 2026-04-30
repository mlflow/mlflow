import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import type { TableOptions } from '@tanstack/react-table';
import { render } from '@testing-library/react';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentEvaluationRunsRowVisibilityProvider } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import {
  EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
  EvalRunsTableKeyedColumnPrefix,
} from './ExperimentEvaluationRunsTable.constants';
import { createEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';

// Capture the columns passed to useReactTable
let capturedTableOptions: TableOptions<any> | undefined;

jest.mock('@databricks/web-shared/react-table', () => {
  const actual = jest.requireActual<typeof import('@databricks/web-shared/react-table')>(
    '@databricks/web-shared/react-table',
  );
  return {
    ...actual,
    useReactTable_unverifiedWithReact18: (_id: string, options: TableOptions<any>) => {
      capturedTableOptions = options;
      return actual.useReactTable_unverifiedWithReact18(_id, options);
    },
  };
});

const metricColumn = createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.METRIC, 'accuracy');
const paramColumn = createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.PARAM, 'model');
const tagColumn = createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.TAG, 'team');

describe('ExperimentEvaluationRunsTable sorting', () => {
  beforeEach(() => {
    capturedTableOptions = undefined;
  });

  test('metric columns use basic (numeric) sorting, param and tag columns use alphanumeric sorting', () => {
    const selectedColumns = {
      ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
      [metricColumn]: true,
      [paramColumn]: true,
      [tagColumn]: true,
    };

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ExperimentEvaluationRunsRowVisibilityProvider>
            <ExperimentEvaluationRunsTable
              data={[]}
              uniqueColumns={[metricColumn, paramColumn, tagColumn]}
              selectedColumns={selectedColumns}
              setSelectedRunUuid={jest.fn()}
              isLoading={false}
              hasNextPage={false}
              rowSelection={{}}
              setRowSelection={jest.fn()}
              setSelectedDatasetWithRun={jest.fn()}
              setIsDrawerOpen={jest.fn()}
              viewMode={ExperimentEvaluationRunsPageMode.TRACES}
            />
          </ExperimentEvaluationRunsRowVisibilityProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(capturedTableOptions).toBeDefined();
    const columns = capturedTableOptions!.columns;

    const metricCol = columns.find((c) => c.id === metricColumn);
    const paramCol = columns.find((c) => c.id === paramColumn);
    const tagCol = columns.find((c) => c.id === tagColumn);

    expect(metricCol).toBeDefined();
    expect(paramCol).toBeDefined();
    expect(tagCol).toBeDefined();

    expect(metricCol!.sortingFn).toBe('basic');
    expect(paramCol!.sortingFn).toBe('alphanumeric');
    expect(tagCol!.sortingFn).toBe('alphanumeric');
  });
});
