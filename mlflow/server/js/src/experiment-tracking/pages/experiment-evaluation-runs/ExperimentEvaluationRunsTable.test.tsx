import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import type { TableOptions } from '@tanstack/react-table';
import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentEvaluationRunsRowVisibilityProvider } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import {
  EVAL_RUNS_COLUMN_SIZES,
  EVAL_RUNS_KEYED_COLUMN_SIZE,
  EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
  EvalRunsTableColumnId,
  EvalRunsTableKeyedColumnPrefix,
  getEvalRunsBaseSelectionState,
} from './ExperimentEvaluationRunsTable.constants';
import { createEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import type { ExperimentEvaluationRunsGroupData, RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import type { RunEntityWithChildren } from '../../types';
import { RunGroupingMode } from '../../components/experiment-page/utils/experimentPage.row-types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../constants';
import Routes from '../../routes';
import { TestRouter, testRoute } from '../../../common/utils/RoutingTestUtils';

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

// RunNameCell pulls run colors from Redux; stub the hooks so we don't need a Provider.
jest.mock('../../components/experiment-page/hooks/useExperimentRunColor', () => ({
  useGetExperimentRunColor: () => () => '#000000',
  useSaveExperimentRunColor: () => jest.fn(),
}));

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

const makeRun = (runUuid: string, extra: Partial<RunEntityWithChildren> = {}): RunEntityWithChildren =>
  ({
    data: { params: [], tags: [], metrics: [] },
    info: {
      artifactUri: '',
      endTime: 0,
      experimentId: 'exp-1',
      lifecycleStage: '',
      runUuid,
      runName: `Run ${runUuid}`,
      startTime: 0,
      status: 'FINISHED',
    },
    inputs: { datasetInputs: [], modelInputs: [] },
    outputs: { modelOutputs: [] },
    ...extra,
  }) as unknown as RunEntityWithChildren;

const withAccuracy = (runUuid: string, value: number) =>
  makeRun(runUuid, {
    data: { params: [], tags: [], metrics: [{ key: 'accuracy', value, step: 0, timestamp: 0 }] },
  } as Partial<RunEntityWithChildren>);

const withDataset = (runUuid: string, name: string, sourceType: string, tags: { key: string; value: string }[] = []) =>
  makeRun(runUuid, {
    inputs: {
      datasetInputs: [
        { dataset: { digest: 'abc123def', name, profile: '', schema: '', source: '{}', sourceType }, tags },
      ],
      modelInputs: [],
    },
  } as Partial<RunEntityWithChildren>);

const renderTable = ({
  data,
  enableNewEvalRunsUi = false,
  uniqueColumns = [],
  selectedColumns,
  isGrouped,
  setSelectedDatasetWithRun = jest.fn(),
  setSelectedRunUuid = jest.fn(),
  setSelectedColumns = jest.fn(),
  initialEntries,
  baselineRun,
  extraRoutes = [],
}: {
  data: RunEntityOrGroupData[];
  enableNewEvalRunsUi?: boolean;
  uniqueColumns?: string[];
  selectedColumns?: { [key: string]: boolean };
  isGrouped?: boolean;
  setSelectedDatasetWithRun?: jest.Mock;
  setSelectedRunUuid?: jest.Mock;
  setSelectedColumns?: jest.Mock;
  initialEntries?: string[];
  baselineRun?: RunEntityWithChildren;
  extraRoutes?: ReturnType<typeof testRoute>[];
}) => {
  const table = (tableData: RunEntityOrGroupData[]) => (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ExperimentEvaluationRunsRowVisibilityProvider>
          <TestRouter
            initialEntries={initialEntries}
            routes={[
              testRoute(
                <ExperimentEvaluationRunsTable
                  data={tableData}
                  {...(baselineRun ? { baselineRun } : {})}
                  uniqueColumns={uniqueColumns}
                  selectedColumns={
                    selectedColumns ?? {
                      ...getEvalRunsBaseSelectionState(enableNewEvalRunsUi),
                      ...Object.fromEntries(uniqueColumns.map((column) => [column, true])),
                    }
                  }
                  setSelectedRunUuid={setSelectedRunUuid}
                  setSelectedColumns={setSelectedColumns}
                  isLoading={false}
                  hasNextPage={false}
                  rowSelection={{}}
                  setRowSelection={jest.fn()}
                  setSelectedDatasetWithRun={setSelectedDatasetWithRun}
                  setIsDrawerOpen={jest.fn()}
                  viewMode={ExperimentEvaluationRunsPageMode.TRACES}
                  isGrouped={isGrouped}
                  enableNewEvalRunsUi={enableNewEvalRunsUi}
                />,
              ),
              ...extraRoutes,
            ]}
          />
        </ExperimentEvaluationRunsRowVisibilityProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
  const result = render(table(data));
  return { ...result, rerenderWithData: (nextData: RunEntityOrGroupData[]) => result.rerender(table(nextData)) };
};

describe('ExperimentEvaluationRunsTable baseline', () => {
  beforeEach(() => {
    capturedTableOptions = undefined;
  });

  test('shows group metric averages without a filtered-out baseline', () => {
    const group: ExperimentEvaluationRunsGroupData = {
      groupKey: 'dataset-a',
      groupValues: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'dataset-a' }],
      subRuns: [withAccuracy('group-1', 0.8), withAccuracy('group-2', 1)],
    };

    renderTable({
      data: [group],
      enableNewEvalRunsUi: true,
      uniqueColumns: [metricColumn],
      isGrouped: true,
      baselineRun: withAccuracy('baseline', 0),
    });

    expect(screen.getByText('BASELINE')).toBeInTheDocument();
    expect(screen.getByText('avg')).toBeInTheDocument();
    expect(screen.getByText('0.9')).toBeInTheDocument();
    expect(screen.queryByText('0.6')).not.toBeInTheDocument();
  });

  test('shows and pins the baseline when filtered results exclude it', () => {
    renderTable({ data: [makeRun('newer')], enableNewEvalRunsUi: true, baselineRun: makeRun('baseline') });

    expect(screen.getByText('BASELINE')).toBeInTheDocument();
    expect(screen.getAllByText(/^Run (baseline|newer)$/).map((node) => node.textContent)).toEqual([
      'Run baseline',
      'Run newer',
    ]);
  });

  test('shows the baseline when filters return no runs', () => {
    renderTable({ data: [], enableNewEvalRunsUi: true, baselineRun: makeRun('baseline') });

    expect(screen.getByText('Run baseline')).toBeInTheDocument();
    expect(screen.getByText('BASELINE')).toBeInTheDocument();
  });

  test('pins the baseline group while keeping the baseline inside it', () => {
    const baseline = makeRun('baseline');
    const baselineGroup: ExperimentEvaluationRunsGroupData = {
      groupKey: 'baseline-group',
      groupValues: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'baseline-dataset' }],
      subRuns: [makeRun('companion'), baseline],
    };
    const otherGroup: ExperimentEvaluationRunsGroupData = {
      groupKey: 'other-group',
      groupValues: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'other-dataset' }],
      subRuns: [makeRun('other')],
    };

    renderTable({
      data: [otherGroup, baselineGroup],
      enableNewEvalRunsUi: true,
      isGrouped: true,
      baselineRun: baseline,
    });

    const pinnedGroup = capturedTableOptions!.data[0] as ExperimentEvaluationRunsGroupData;
    expect(pinnedGroup.groupKey).toBe('baseline-group');
    expect(pinnedGroup.subRuns.map((run) => run.info.runUuid)).toEqual(['companion', 'baseline']);
    expect(capturedTableOptions!.data).toHaveLength(2);

    const rows = screen.getAllByRole('row');
    const baselineGroupIndex = rows.findIndex((row) => row.textContent?.includes('baseline-dataset'));
    const baselineIndex = rows.findIndex((row) => row.textContent?.includes('Run baseline'));
    const otherGroupIndex = rows.findIndex((row) => row.textContent?.includes('other-dataset'));
    expect(baselineGroupIndex).toBeGreaterThan(0);
    expect(baselineIndex).toBeGreaterThan(baselineGroupIndex);
    expect(baselineIndex).toBeLessThan(otherGroupIndex);
    expect(screen.getByText('BASELINE')).toBeInTheDocument();
  });

  test('keeps an expanded baseline subtree together when sorting', () => {
    const baseline = makeRun('parent', { children: [makeRun('child')] });
    renderTable({ data: [makeRun('aaa'), baseline], enableNewEvalRunsUi: true, baselineRun: makeRun('parent') });

    act(() => capturedTableOptions!.onSortingChange?.([{ id: EvalRunsTableColumnId.runName, desc: false }]));

    expect(screen.getAllByText(/^Run (parent|child|aaa)$/).map((node) => node.textContent)).toEqual([
      'Run parent',
      'Run child',
      'Run aaa',
    ]);
  });

  test('pins nothing and shows no BASELINE pill with the flag off', () => {
    renderTable({ data: [makeRun('aaa'), makeRun('bbb')], baselineRun: makeRun('bbb') });

    expect(screen.queryByText('BASELINE')).not.toBeInTheDocument();
    expect(screen.getAllByText(/^Run (aaa|bbb)$/).map((node) => node.textContent)).toEqual(['Run aaa', 'Run bbb']);
    expect(capturedTableOptions!.meta).not.toHaveProperty('baselineRunUuid');
  });
});

describe('ExperimentEvaluationRunsTable metric cells', () => {
  test('rounds metric values to four decimal places in the cell and shows the full value in a tooltip', async () => {
    renderTable({
      data: [withAccuracy('run-1', 0.123456789)],
      uniqueColumns: [metricColumn],
      enableNewEvalRunsUi: true,
    });

    const cell = screen.getByText('0.1235');
    expect(cell).not.toHaveAttribute('title');
    await userEvent.hover(cell);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('0.123456789');
  });

  test('flag off: metric values keep their full precision', () => {
    renderTable({ data: [withAccuracy('run-1', 0.123456789)], uniqueColumns: [metricColumn] });

    expect(screen.getByText('0.123456789')).toHaveAttribute('title', '0.123456789');
    expect(screen.queryByText('0.1235')).not.toBeInTheDocument();
  });
});

describe('ExperimentEvaluationRunsTable columns', () => {
  test('flag on: the Dataset version cell renders the input dataset digest', () => {
    renderTable({
      data: [withDataset('run-1', 'my_dataset', 'code')],
      enableNewEvalRunsUi: true,
      selectedColumns: { ...getEvalRunsBaseSelectionState(true), [EvalRunsTableColumnId.datasetVersion]: true },
    });

    expect(screen.getByRole('columnheader', { name: 'Dataset version' })).toBeInTheDocument();
    expect(screen.getByText('abc123def')).toBeInTheDocument();
  });

  test('flag on: Dataset version and Status are hidden by default', () => {
    expect(getEvalRunsBaseSelectionState(true)).toMatchObject({
      [EvalRunsTableColumnId.datasetVersion]: false,
      [EvalRunsTableColumnId.status]: false,
    });

    renderTable({ data: [withDataset('run-1', 'my_dataset', 'code')], enableNewEvalRunsUi: true });

    expect(screen.queryByRole('columnheader', { name: 'Dataset version' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Status' })).not.toBeInTheDocument();
  });

  test('flag on: the Status column can be shown and sorts runs by status', () => {
    const withStatus = (runUuid: string, status: string) => {
      const run = makeRun(runUuid);
      return { ...run, info: { ...run.info, status } } as RunEntityWithChildren;
    };
    renderTable({
      data: [withStatus('running', 'RUNNING'), withStatus('failed', 'FAILED'), withStatus('finished', 'FINISHED')],
      enableNewEvalRunsUi: true,
      selectedColumns: { ...getEvalRunsBaseSelectionState(true), [EvalRunsTableColumnId.status]: true },
    });

    expect(screen.getByRole('columnheader', { name: 'Status' })).toBeInTheDocument();
    act(() => capturedTableOptions!.onSortingChange?.([{ id: EvalRunsTableColumnId.status, desc: false }]));

    expect(screen.getAllByText(/^Run (running|failed|finished)$/).map((node) => node.textContent)).toEqual([
      'Run failed',
      'Run finished',
      'Run running',
    ]);
  });

  test('flag off: the column defaults and the Status column are unchanged, and Dataset version is never rendered', () => {
    expect(getEvalRunsBaseSelectionState(false)).toBe(EVAL_RUNS_TABLE_BASE_SELECTION_STATE);

    renderTable({
      data: [withDataset('run-1', 'my_dataset', 'code')],
      selectedColumns: { ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE, [EvalRunsTableColumnId.datasetVersion]: true },
    });

    expect(screen.getByRole('columnheader', { name: 'Status' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Dataset version' })).not.toBeInTheDocument();
    expect(screen.queryByText('abc123def')).not.toBeInTheDocument();
  });
});

describe('ExperimentEvaluationRunsTable run name cell', () => {
  test('nested-run expand button renders for parent runs with child runs only', () => {
    renderTable({ data: [makeRun('parent-1', { children: [makeRun('child-1')] }), makeRun('leaf-1')] });

    expect(screen.getAllByRole('button', { name: 'Collapse child runs' })).toHaveLength(1);
  });

  test('shows the full run name on hover when the cell is truncated', async () => {
    const scrollWidthSpy = jest.spyOn(HTMLElement.prototype, 'scrollWidth', 'get').mockReturnValue(300);
    const clientWidthSpy = jest.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(120);
    const longRunName = 'New Evals — Release Readiness Check With A Long Name';
    const run = makeRun('long-name');
    run.info.runName = longRunName;

    try {
      renderTable({ data: [run], enableNewEvalRunsUi: true });
      await userEvent.hover(screen.getByText(longRunName));

      expect(await screen.findByRole('tooltip')).toHaveTextContent(longRunName);
    } finally {
      scrollWidthSpy.mockRestore();
      clientWidthSpy.mockRestore();
    }
  });

  test('preserves the legacy run-name cell without a truncation tooltip', async () => {
    const scrollWidthSpy = jest.spyOn(HTMLElement.prototype, 'scrollWidth', 'get').mockReturnValue(300);
    const clientWidthSpy = jest.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(120);
    const longRunName = 'Legacy Evaluation Run With A Long Name';
    const run = makeRun('legacy-long-name');
    run.info.runName = longRunName;

    try {
      renderTable({ data: [run] });
      await userEvent.hover(screen.getByText(longRunName));

      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    } finally {
      scrollWidthSpy.mockRestore();
      clientWidthSpy.mockRestore();
    }
  });

  test('flag on: reserves a chevron slot on leaf rows when any row can expand', () => {
    const { container } = renderTable({
      data: [makeRun('parent-1', { children: [makeRun('child-1')] })],
      enableNewEvalRunsUi: true,
    });

    // eslint-disable-next-line testing-library/no-container, testing-library/no-node-access -- the slot is aria-hidden
    // The filler cells are aria-hidden too, but they carry a cell role.
    expect(container.querySelectorAll('div[aria-hidden="true"]:not([role])')).toHaveLength(1);
  });

  test('flag off: leaf rows get no chevron slot', () => {
    const { container } = renderTable({ data: [makeRun('parent-1', { children: [makeRun('child-1')] })] });

    // eslint-disable-next-line testing-library/no-container, testing-library/no-node-access -- the slot is aria-hidden
    expect(container.querySelectorAll('div[aria-hidden="true"]')).toHaveLength(0);
  });

  test('flag on: the sticky header row stacks above row checkboxes scrolled under it', () => {
    renderTable({ data: [makeRun('run-1')], enableNewEvalRunsUi: true });

    const headerRow = screen.getAllByRole('row')[0];
    expect(within(headerRow).getByRole('checkbox', { name: 'Select all runs' })).toBeInTheDocument();
    expect(headerRow).toHaveStyle({ zIndex: '2' });
  });

  test('flag off: the header row keeps the default stacking', () => {
    renderTable({ data: [makeRun('run-1')] });

    expect(screen.getAllByRole('row')[0].style.zIndex).toBe('');
  });

  test('flag on: issue-detection run names are real links to the issue-detection page', () => {
    const run = makeRun('issue-run', {
      data: {
        params: [],
        metrics: [],
        tags: [{ key: MLFLOW_RUN_TYPE_TAG, value: MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION }],
      },
    } as Partial<RunEntityWithChildren>);

    renderTable({ data: [run, makeRun('eval-run')], enableNewEvalRunsUi: true });

    expect(screen.getByRole('link', { name: 'Run issue-run' })).toHaveAttribute(
      'href',
      Routes.getIssueDetectionRunDetailsRoute('exp-1', 'issue-run'),
    );
  });

  test('flag on: eval run names are real links to the evaluation run details page', () => {
    renderTable({ data: [makeRun('eval-run')], enableNewEvalRunsUi: true });

    expect(screen.getByRole('link', { name: 'Run eval-run' })).toHaveAttribute(
      'href',
      Routes.getEvaluationRunDetailsRoute('exp-1', 'eval-run'),
    );
  });

  test('flag off: eval run names are not links', () => {
    renderTable({ data: [makeRun('eval-run')] });

    expect(screen.queryByRole('link', { name: 'Run eval-run' })).not.toBeInTheDocument();
    expect(screen.getByText('Run eval-run')).toBeInTheDocument();
  });

  test('flag on: issue-detection run links keep the time range query', () => {
    const run = makeRun('issue-run', {
      data: {
        params: [],
        metrics: [],
        tags: [{ key: MLFLOW_RUN_TYPE_TAG, value: MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION }],
      },
    } as Partial<RunEntityWithChildren>);

    renderTable({
      data: [run],
      enableNewEvalRunsUi: true,
      initialEntries: ['/?startTimeLabel=CUSTOM&startTime=100&endTime=200&unrelated=1'],
    });

    expect(screen.getByRole('link', { name: 'Run issue-run' })).toHaveAttribute(
      'href',
      `${Routes.getIssueDetectionRunDetailsRoute('exp-1', 'issue-run')}?startTimeLabel=CUSTOM&startTime=100&endTime=200`,
    );
  });

  test('flag on: clicking an eval run name opens the evaluation run details page', async () => {
    const setSelectedRunUuid = jest.fn();
    renderTable({
      data: [makeRun('eval-run')],
      enableNewEvalRunsUi: true,
      setSelectedRunUuid,
      initialEntries: ['/experiments/exp-1/evaluation-runs'],
      extraRoutes: [
        testRoute(<div>Evaluation run details</div>, '/experiments/:experimentId/evaluation-runs/:runUuid/details'),
      ],
    });

    await userEvent.click(screen.getByText('Run eval-run'));

    expect(await screen.findByText('Evaluation run details')).toBeInTheDocument();
    expect(setSelectedRunUuid).not.toHaveBeenCalled();
  });

  test('flag off: clicking an eval run name opens the run in the split view', async () => {
    const setSelectedRunUuid = jest.fn();
    renderTable({ data: [makeRun('eval-run')], setSelectedRunUuid });

    await userEvent.click(screen.getByText('Run eval-run'));

    expect(setSelectedRunUuid).toHaveBeenCalledWith('eval-run');
  });

  test('flag off: issue-detection run names are not links', () => {
    const run = makeRun('issue-run', {
      data: {
        params: [],
        metrics: [],
        tags: [{ key: MLFLOW_RUN_TYPE_TAG, value: MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION }],
      },
    } as Partial<RunEntityWithChildren>);

    renderTable({ data: [run] });

    expect(screen.queryByRole('link', { name: 'Run issue-run' })).not.toBeInTheDocument();
    expect(screen.getByText('Run issue-run')).toBeInTheDocument();
  });

  test('flag on: the row checkbox has an accessible label', () => {
    renderTable({ data: [makeRun('run-1')], enableNewEvalRunsUi: true });

    expect(screen.getByRole('checkbox', { name: 'Select run Run run-1' })).toBeInTheDocument();
  });

  test('flag off: the row checkbox has no aria-label', () => {
    renderTable({ data: [makeRun('run-1')] });

    expect(screen.getByRole('checkbox')).not.toHaveAttribute('aria-label');
  });
});

describe('ExperimentEvaluationRunsTable group rows', () => {
  const group: ExperimentEvaluationRunsGroupData = {
    groupKey: 'dataset-a',
    groupValues: [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: 'dataset-a' }],
    subRuns: [withAccuracy('group-1', 0.8), withAccuracy('group-2', 1)],
  };

  test('shows group metric averages in a secondary group row', () => {
    const { rerenderWithData } = renderTable({
      data: [group],
      uniqueColumns: [metricColumn],
      isGrouped: true,
      enableNewEvalRunsUi: true,
    });

    const groupRow = screen.getByRole('row', { name: /dataset-a/ });
    expect(groupRow).toHaveAttribute('style', expect.stringContaining('background-color'));
    expect(screen.getByText('avg')).toBeInTheDocument();
    expect(screen.getByText('0.9')).toBeInTheDocument();

    rerenderWithData([{ ...group, subRuns: [withAccuracy('group-1', 0.6), withAccuracy('group-2', 0.8)] }]);
    expect(screen.getByText('0.7')).toBeInTheDocument();
  });

  test('gives the group label cell an accessible span up to the first metric column', () => {
    renderTable({ data: [group], uniqueColumns: [metricColumn], isGrouped: true, enableNewEvalRunsUi: true });

    const headers = screen.getAllByRole('columnheader');
    const runNameIndex = headers.findIndex((header) => /Run Name/.test(header.textContent ?? ''));
    const metricIndex = headers.findIndex((header) => /accuracy/.test(header.textContent ?? ''));
    const labelCell = screen.getByText('dataset-a').closest('[role="cell"]');
    expect(labelCell).toHaveAttribute('aria-colspan', String(metricIndex - runNameIndex));
  });

  test('shows every lineage group tag, with the full value in a tooltip', async () => {
    const user = userEvent.setup();
    const digest = 'sha256:4c16af46ac51d2a964f7d0b41e3707a09a22fc3e46eaeba217cb47073d255759';
    const lineageGroup: ExperimentEvaluationRunsGroupData = {
      groupKey: 'lineage-group',
      groupValues: [
        { mode: RunGroupingMode.Tag, groupByData: 'mlflow.genaiEvaluate.scorersDigest', value: digest },
        { mode: RunGroupingMode.Tag, groupByData: 'mlflow.genaiEvaluate.agent', value: '__main__.answer_question_v2' },
      ],
      subRuns: [makeRun('grouped')],
    };

    renderTable({ data: [lineageGroup], isGrouped: true, enableNewEvalRunsUi: true });

    expect(screen.getByText('__main__.answer_question_v2', { exact: false })).toBeInTheDocument();
    // The label replaces the empty cells after the run name, so the group row has fewer cells.
    const rows = screen.getAllByRole('row');
    const groupRow = rows.find((row) => row.textContent?.includes('Group:'));
    const runRow = rows.find((row) => row.textContent?.includes('Run grouped'));
    expect(groupRow && runRow).toBeTruthy();
    expect(within(groupRow!).getAllByRole('cell').length).toBeLessThan(within(runRow!).getAllByRole('cell').length);
    await user.hover(screen.getByText('4c16af4'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(`Scorers: ${digest}`);
  });

  test('labels a partial scorers group with its counts', async () => {
    const user = userEvent.setup();
    const digest = 'sha256:77e63932b4d7109dea4ee21ff10b7674f776276096e2db0548db047ea68adb77';
    const partialGroup: ExperimentEvaluationRunsGroupData = {
      groupKey: 'partial-group',
      groupValues: [
        {
          mode: RunGroupingMode.Tag,
          groupByData: 'mlflow.genaiEvaluate.scorersDigest',
          value: `${digest} partial:2/3`,
        },
      ],
      subRuns: [makeRun('grouped')],
    };

    renderTable({ data: [partialGroup], isGrouped: true, enableNewEvalRunsUi: true });

    const label = screen.getByText('77e6393 (partial: 2 of 3)', { exact: false });
    await user.hover(label);
    expect(await screen.findByRole('tooltip')).toHaveTextContent(`Scorers: ${digest} (partial: 2 of 3)`);
  });

  test('shows the full group average in a tooltip', async () => {
    renderTable({
      data: [{ ...group, subRuns: [withAccuracy('group-1', 0.1), withAccuracy('group-2', 0.2)] }],
      uniqueColumns: [metricColumn],
      isGrouped: true,
      enableNewEvalRunsUi: true,
    });

    await userEvent.hover(screen.getByText('0.15'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(String((0.1 + 0.2) / 2));
  });

  test('flag off: group rows keep the legacy label-only row without a background', () => {
    renderTable({ data: [group], uniqueColumns: [metricColumn], isGrouped: true });

    const groupRow = screen.getByRole('row', { name: /dataset-a/ });
    expect(groupRow).not.toHaveAttribute('style', expect.stringContaining('background-color'));
    expect(screen.queryByText('avg')).not.toBeInTheDocument();
    expect(screen.queryByText('0.9')).not.toBeInTheDocument();
  });
});

describe('ExperimentEvaluationRunsTable dataset cell', () => {
  test('labels an unnamed (default-named) code dataset "Unnamed dataset"', () => {
    renderTable({ data: [withDataset('run-1', 'dataset', 'code')], enableNewEvalRunsUi: true });
    expect(screen.getByText('Unnamed dataset')).toBeInTheDocument();
    // The generic placeholder name should no longer appear (the "Dataset" column header is distinct).
    expect(screen.queryByText('dataset')).not.toBeInTheDocument();
  });

  test('labels an unnamed local-file dataset "Unnamed dataset"', () => {
    renderTable({ data: [withDataset('run-1', 'dataset', 'local')], enableNewEvalRunsUi: true });
    expect(screen.getByText('Unnamed dataset')).toBeInTheDocument();
  });

  test('treats any default-named dataset as unnamed regardless of source type', () => {
    // Detection is name-based: even an s3 dataset logged with the generic "dataset" name is unnamed.
    renderTable({ data: [withDataset('run-1', 'dataset', 's3')], enableNewEvalRunsUi: true });
    expect(screen.getByText('Unnamed dataset')).toBeInTheDocument();
    expect(screen.queryByText('dataset')).not.toBeInTheDocument();
  });

  test('shows the real name for a named dataset (not relabeled)', async () => {
    renderTable({ data: [withDataset('run-1', 'my-bucket-data', 's3')], enableNewEvalRunsUi: true });
    expect(screen.getByText('my-bucket-data')).toBeInTheDocument();
    expect(screen.queryByText('Unnamed dataset')).not.toBeInTheDocument();
    // The new-UI hover card pairs the real name with its digest and source type.
    await userEvent.hover(screen.getByText('my-bucket-data'));
    expect(await screen.findByText('abc123def')).toBeInTheDocument();
    expect(screen.getByText('s3')).toBeInTheDocument();
  });

  test('shows the real name even for a named code/local dataset', async () => {
    renderTable({ data: [withDataset('run-1', 'qa_smoke_set', 'code')], enableNewEvalRunsUi: true });
    expect(screen.getByText('qa_smoke_set')).toBeInTheDocument();
    expect(screen.queryByText('Unnamed dataset')).not.toBeInTheDocument();
    await userEvent.hover(screen.getByText('qa_smoke_set'));
    expect(await screen.findByText('abc123def')).toBeInTheDocument();
    expect(screen.getByText('code')).toBeInTheDocument();
    // Datasets that weren't logged from an evaluation dataset have nothing to open.
    expect(screen.queryByRole('link', { name: /View dataset/ })).not.toBeInTheDocument();
  });

  test('links to the evaluation dataset from the hover card when the input came from one', async () => {
    const run = makeRun('run-1', {
      inputs: {
        datasetInputs: [
          {
            dataset: {
              digest: 'd1',
              name: 'golden-set',
              profile: '',
              schema: '',
              source: JSON.stringify({ dataset_id: 'd-123' }),
              sourceType: 'mlflow_evaluation_dataset',
            },
            tags: [],
          },
        ],
        modelInputs: [],
      },
    } as Partial<RunEntityWithChildren>);
    renderTable({ data: [run], enableNewEvalRunsUi: true });

    await userEvent.hover(screen.getByText('golden-set'));
    const link = await screen.findByRole('link', { name: /View dataset/ });
    expect(link).toHaveAttribute('href', Routes.getExperimentPageDatasetDetailRoute('exp-1', 'd-123'));
    expect(link).toHaveAttribute('target', '_blank');
  });

  test('flag off: the dataset pill keeps the name-only tooltip', async () => {
    renderTable({ data: [withDataset('run-1', 'qa_smoke_set', 'code')] });

    await userEvent.hover(screen.getByText('qa_smoke_set'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(/^qa_smoke_set$/);
    expect(screen.queryByText('abc123def')).not.toBeInTheDocument();
  });

  test('flag off: a default-named dataset keeps its raw name', () => {
    renderTable({ data: [withDataset('run-1', 'dataset', 'code')] });
    expect(screen.getByText('dataset')).toBeInTheDocument();
    expect(screen.queryByText('Unnamed dataset')).not.toBeInTheDocument();
  });

  test('flag on: opening the drawer passes the dataset input tags', async () => {
    const setSelectedDatasetWithRun = jest.fn();
    const tags = [{ key: 'mlflow.data.context', value: 'eval' }];
    renderTable({
      data: [withDataset('run-1', 'my_dataset', 'code', tags)],
      enableNewEvalRunsUi: true,
      setSelectedDatasetWithRun,
    });

    await userEvent.click(screen.getByText('my_dataset'));

    expect(setSelectedDatasetWithRun).toHaveBeenCalledWith(
      expect.objectContaining({ datasetWithTags: expect.objectContaining({ tags }) }),
    );
  });

  test('flag off: opening the drawer passes the dataset without its tags', async () => {
    const setSelectedDatasetWithRun = jest.fn();
    renderTable({
      data: [withDataset('run-1', 'my_dataset', 'code', [{ key: 'mlflow.data.context', value: 'eval' }])],
      setSelectedDatasetWithRun,
    });

    await userEvent.click(screen.getByText('my_dataset'));

    const [{ datasetWithTags }] = setSelectedDatasetWithRun.mock.calls[0] as [
      { datasetWithTags: Record<string, unknown> },
    ];
    expect(datasetWithTags).not.toHaveProperty('tags');
  });
});

const getRunNameOrder = () => screen.getAllByText(/^Run run-[abc]$/).map((element) => element.textContent);

describe('ExperimentEvaluationRunsTable header menu', () => {
  test('flag on: sorts a column ascending and descending from its header menu', async () => {
    renderTable({ data: [makeRun('run-b'), makeRun('run-a'), makeRun('run-c')], enableNewEvalRunsUi: true });

    await userEvent.click(screen.getByRole('button', { name: 'Options for Run Name column' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Sort ascending' }));
    expect(getRunNameOrder()).toEqual(['Run run-a', 'Run run-b', 'Run run-c']);

    await userEvent.click(screen.getByRole('button', { name: 'Options for Run Name column' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Sort descending' }));
    expect(getRunNameOrder()).toEqual(['Run run-c', 'Run run-b', 'Run run-a']);
  });

  test('flag on: hides a column from its header menu', async () => {
    const setSelectedColumns = jest.fn();
    renderTable({ data: [makeRun('run-a')], enableNewEvalRunsUi: true, setSelectedColumns });

    await userEvent.click(screen.getByRole('button', { name: 'Options for Dataset column' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Hide column' }));

    expect(setSelectedColumns).toHaveBeenCalledWith(expect.objectContaining({ dataset: false, run_name: true }));
  });

  test('flag on: the run name column can be sorted but not hidden', async () => {
    renderTable({ data: [makeRun('run-a')], enableNewEvalRunsUi: true });

    await userEvent.click(screen.getByRole('button', { name: 'Options for Run Name column' }));

    expect(await screen.findByRole('menuitem', { name: 'Sort ascending' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Hide column' })).not.toBeInTheDocument();
  });

  test('flag on: run name, dataset and dataset version columns are sortable', () => {
    renderTable({
      data: [makeRun('run-a')],
      enableNewEvalRunsUi: true,
      selectedColumns: { ...getEvalRunsBaseSelectionState(true), [EvalRunsTableColumnId.datasetVersion]: true },
    });

    const sortableColumnIds = capturedTableOptions!.columns
      .filter((column) => column.enableSorting && 'accessorFn' in column)
      .map((column) => column.id);
    expect(sortableColumnIds).toEqual(
      expect.arrayContaining([
        EvalRunsTableColumnId.runName,
        EvalRunsTableColumnId.dataset,
        EvalRunsTableColumnId.datasetVersion,
      ]),
    );
  });

  test('flag on: shows the active sort direction as a toggle in the header', async () => {
    renderTable({ data: [makeRun('run-b'), makeRun('run-a')], enableNewEvalRunsUi: true });

    await userEvent.click(screen.getByRole('button', { name: 'Options for Run Name column' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Sort ascending' }));
    await userEvent.click(screen.getByRole('button', { name: 'Toggle sort direction for Run Name' }));

    expect(getRunNameOrder()).toEqual(['Run run-b', 'Run run-a']);
  });

  test('flag off: headers have no options menu and the run name column is not sortable', () => {
    renderTable({ data: [makeRun('run-b'), makeRun('run-a')] });

    expect(screen.queryByRole('button', { name: /^Options for .* column$/ })).not.toBeInTheDocument();
    const runNameColumn = capturedTableOptions!.columns.find((column) => column.id === EvalRunsTableColumnId.runName);
    expect(runNameColumn).not.toHaveProperty('accessorFn');
    expect(getRunNameOrder()).toEqual(['Run run-b', 'Run run-a']);
  });
});

describe('ExperimentEvaluationRunsTable column sizing', () => {
  const COLUMN_SIZING_STORAGE_KEY = 'mlflow.eval-runs.column-sizing_v2';

  beforeEach(() => {
    localStorage.clear();
  });

  test('flag on: columns get resizable widths, seeded from the persisted sizes', () => {
    localStorage.setItem(COLUMN_SIZING_STORAGE_KEY, JSON.stringify({ [EvalRunsTableColumnId.runName]: 333 }));

    const { container } = renderTable({ data: [makeRun('run-a')], enableNewEvalRunsUi: true });

    const tableWithSizes = container.querySelector<HTMLElement>('[style*="--eval-runs-table-row-width"]');
    expect(tableWithSizes).not.toBeNull();
    // The checkbox column is index 0; Run Name follows it.
    expect(tableWithSizes!.style.getPropertyValue('--eval-runs-table-column-1')).toBe('333px');
    expect(tableWithSizes!.style.getPropertyValue('--eval-runs-table-column-2')).toBe('100px');
  });

  test('flag on: columns become resizable and are seeded with shared size specs', () => {
    renderTable({
      data: [makeRun('run-a')],
      enableNewEvalRunsUi: true,
      uniqueColumns: [metricColumn],
      selectedColumns: {
        ...getEvalRunsBaseSelectionState(true),
        [EvalRunsTableColumnId.datasetVersion]: true,
        [metricColumn]: true,
      },
    });

    expect(capturedTableOptions!.enableColumnResizing).toBe(true);
    const getColumn = (id: string) => capturedTableOptions!.columns.find((column) => column.id === id);
    expect(getColumn(EvalRunsTableColumnId.runName)?.size).toBe(
      EVAL_RUNS_COLUMN_SIZES[EvalRunsTableColumnId.runName]?.size,
    );
    expect(getColumn(metricColumn)?.size).toBe(EVAL_RUNS_KEYED_COLUMN_SIZE.size);
    expect(getColumn(EvalRunsTableColumnId.checkbox)?.size).toBe(32);
    expect(getColumn(EvalRunsTableColumnId.datasetVersion)?.size).toBe(
      EVAL_RUNS_COLUMN_SIZES[EvalRunsTableColumnId.datasetVersion]?.size,
    );
  });

  test('flag on: widening a column changes no size or offset of the columns to its left', async () => {
    const { container } = renderTable({
      data: [makeRun('run-a')],
      enableNewEvalRunsUi: true,
      uniqueColumns: [metricColumn],
    });
    const sizes = () => {
      const element = container.querySelector<HTMLElement>('[style*="--eval-runs-table-row-width"]')!;
      return Array.from({ length: 8 }, (_, index) =>
        element.style.getPropertyValue(`--eval-runs-table-column-${index}`),
      );
    };
    const headers = screen.getAllByRole('columnheader');
    const datasetIndex = headers.findIndex((header) => header.textContent?.startsWith('Dataset'));
    // Each column is pinned to its own width variable, so a column's offset is the sum of the widths
    // to its left: if no column grows, a resize can only move the resized column's right edge.
    headers.forEach((header) => {
      expect(header.style.flexGrow).toBe('0');
    });
    const before = sizes();

    await userEvent.click(within(headers[datasetIndex]).getByRole('button', { name: 'Resize Column' }));
    const [, increaseWidth] = within(await screen.findByRole('dialog')).getAllByRole('button');
    await userEvent.click(increaseWidth);

    const after = sizes();
    expect(after.slice(0, datasetIndex)).toEqual(before.slice(0, datasetIndex));
    expect(parseInt(after[datasetIndex], 10)).toBe(parseInt(before[datasetIndex], 10) + 10);
  });

  test('flag on: leftover width goes to an empty, hidden filler cell in the header and every row', () => {
    const { container } = renderTable({
      data: [makeRun('run-a'), makeRun('run-b')],
      enableNewEvalRunsUi: true,
    });

    const rows = container.querySelectorAll('[role="row"]');
    rows.forEach((row) => {
      const filler = row.lastElementChild as HTMLElement;
      expect(filler).toHaveAttribute('aria-hidden', 'true');
      expect(filler.style.flexGrow).toBe('1');
      expect(filler.textContent).toBe('');
    });
    expect(screen.getAllByRole('columnheader').every((header) => header.getAttribute('aria-hidden') !== 'true')).toBe(
      true,
    );
  });

  test('flag off: columns keep their fixed widths and ignore persisted sizes', () => {
    localStorage.setItem(COLUMN_SIZING_STORAGE_KEY, JSON.stringify({ [EvalRunsTableColumnId.runName]: 333 }));
    const getItemSpy = jest.spyOn(window.localStorage, 'getItem');
    getItemSpy.mockClear();

    const { container } = renderTable({ data: [makeRun('run-a')] });

    expect(getItemSpy).not.toHaveBeenCalledWith(COLUMN_SIZING_STORAGE_KEY);
    getItemSpy.mockRestore();

    expect(container.querySelector('[style*="--eval-runs-table-"]')).toBeNull();
    expect(capturedTableOptions!.enableColumnResizing).toBe(false);
    expect(capturedTableOptions!.initialState).toBeUndefined();
    expect(
      capturedTableOptions!.columns.find((column) => column.id === EvalRunsTableColumnId.runName)?.size,
    ).toBeUndefined();
  });
});
