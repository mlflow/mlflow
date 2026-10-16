import { describe, expect, jest, test } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createIntl } from 'react-intl';
import {
  convertTraceInfoV3ToRunEvalEntry,
  getAssessmentInfos,
  type AssessmentFilter,
} from '@databricks/web-shared/genai-traces-table';
import { getAssessmentAggregates } from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { TracesTable, type TracesTableProps, type TraceTableColumn } from '@databricks/web-shared/traces-table';
import { renderWithProviders } from '@databricks/web-shared/traces-table/test-utils/renderWithProviders';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { buildAssessmentColumnDefs } from '../../components/experiment-page/components/traces-v4/utils/buildAssessmentColumnDefs';
import {
  buildEvalRunAssessmentColumnDefs,
  EvalRunAssessmentHeaderToggle,
  hasCollapsibleAssessmentChart,
} from './EvalRunAssessmentColumnHeader';
import type { EvalRunAssessmentAggregate } from './useEvalRunAssessmentAggregates';

const RUN_UUID = 'run-1';

const TRACES: ModelTraceInfoV3[] = [
  makeTrace('tr-1', {
    assessments: [
      makeFeedbackAssessment('correctness', 'yes'),
      makeFeedbackAssessment('is_polite', true),
      makeFeedbackAssessment('relevance', 4),
    ],
  }),
  makeTrace('tr-2', {
    assessments: [
      makeFeedbackAssessment('correctness', 'yes'),
      makeFeedbackAssessment('is_polite', false),
      makeFeedbackAssessment('relevance', 2),
    ],
  }),
  makeTrace('tr-3', {
    assessments: [makeFeedbackAssessment('correctness', 'no'), makeFeedbackAssessment('is_polite', true)],
  }),
  makeTrace('tr-4', { assessments: [makeFeedbackAssessment('correctness', 'yes')] }),
];

/** The same summaries the page computes, from fixture traces. */
const summarize = (traces: ModelTraceInfoV3[]): Map<string, EvalRunAssessmentAggregate> => {
  const results = traces.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
  const entries = results.map((currentRunValue) => ({ currentRunValue }));
  const infos = getAssessmentInfos(createIntl({ locale: 'en' }), results, undefined);
  return new Map(
    infos.map((info) => [
      info.name,
      {
        name: info.name,
        displayName: info.displayName,
        dtype: info.dtype,
        assessmentInfo: info,
        aggregates: getAssessmentAggregates(info, entries, []),
      },
    ]),
  );
};

const baseColumns = buildAssessmentColumnDefs([
  { name: 'correctness', type: 'categorical' },
  { name: 'is_polite', type: 'categorical' },
  { name: 'relevance', type: 'numeric' },
  { name: 'not_summarized', type: 'categorical' },
]);

const tableProps = (extraColumns: TraceTableColumn[], over: Partial<TracesTableProps> = {}): TracesTableProps => ({
  traces: TRACES,
  visibleColumns: ['input'],
  extraColumns,
  initialColumnSizing: {},
  onColumnSizingSettled: jest.fn(),
  isLoading: false,
  isFetching: false,
  skeletonRowCount: 25,
  onTraceSelected: jest.fn(),
  selectedForBulk: new Map(),
  isAllOnPageSelected: false,
  isSomeOnPageSelected: false,
  onToggleBulkRow: jest.fn(),
  onToggleBulkAll: jest.fn(),
  sort: 'start_time',
  dir: 'desc',
  onSort: jest.fn(),
  onHideColumn: jest.fn(),
  ...over,
});

const build = ({
  assessmentFilters = [],
  toggleAssessmentFilter = jest.fn(),
  areChartsCollapsed = false,
}: {
  assessmentFilters?: AssessmentFilter[];
  toggleAssessmentFilter?: (name: string, value: unknown) => void;
  areChartsCollapsed?: boolean;
} = {}) =>
  buildEvalRunAssessmentColumnDefs({
    columnDefs: baseColumns,
    scorersByName: summarize(TRACES),
    assessmentFilters,
    toggleAssessmentFilter,
    runUuid: RUN_UUID,
    areChartsCollapsed,
  });

const header = (name: RegExp) => screen.getByRole('columnheader', { name });

describe('evaluation run assessment column headers', () => {
  test('gives summarized assessment columns a rich summary header and leaves the rest as they are', () => {
    const columns = build();
    const byId = new Map(columns.map((column) => [column.id, column]));

    expect(byId.get('assessment:correctness')?.headerVariant).toBe('rich');
    expect(byId.get('assessment:relevance')?.headerVariant).toBe('rich');
    expect(byId.get('assessment:not_summarized')).toBe(baseColumns[3]);
    expect(hasCollapsibleAssessmentChart(columns)).toBe(true);
    expect(hasCollapsibleAssessmentChart(baseColumns)).toBe(false);
  });

  test('shows the pass rate, true rate and average, with charts for pass/fail and boolean scores', async () => {
    await renderWithProviders(<TracesTable {...tableProps(build())} />);

    const correctness = header(/^Correctness/);
    expect(correctness).toHaveAccessibleName(/^Correctness PASS 75% /);
    expect(within(correctness).getByRole('progressbar', { name: 'Pass' })).toHaveAttribute('aria-valuenow', '75');

    const polite = header(/^is_polite/);
    expect(polite).toHaveAccessibleName(/^is_polite TRUE 67% /);
    expect(within(polite).getAllByRole('progressbar')).toHaveLength(2);

    const relevance = header(/^relevance/);
    expect(relevance).toHaveAccessibleName('relevance AVG 3');
    // Numeric scores show only their average.
    expect(within(relevance).queryByRole('progressbar')).not.toBeInTheDocument();
  });

  test('hides the charts while collapsed and keeps the summary values', async () => {
    await renderWithProviders(<TracesTable {...tableProps(build({ areChartsCollapsed: true }))} />);

    const correctness = header(/^Correctness/);
    expect(correctness).toHaveAccessibleName('Correctness PASS 75%');
    expect(within(correctness).queryByRole('progressbar')).not.toBeInTheDocument();
  });

  test('toggles a filter on the value of a clicked chart row', async () => {
    const toggleAssessmentFilter = jest.fn();
    await renderWithProviders(<TracesTable {...tableProps(build({ toggleAssessmentFilter }))} />);

    await userEvent.click(within(header(/^Correctness/)).getByRole('progressbar', { name: 'Fail' }));

    expect(toggleAssessmentFilter).toHaveBeenCalledWith('correctness', 'no');
  });

  test('renders the collapse toggle over the header and reports its state', async () => {
    const onToggle = jest.fn();
    await renderWithProviders(
      <TracesTable
        {...tableProps(build(), {
          headerBottomOverlay: <EvalRunAssessmentHeaderToggle areChartsCollapsed={false} onToggle={onToggle} />,
        })}
      />,
    );

    const toggle = screen.getByRole('button', { name: 'Collapse assessment charts' });
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await userEvent.click(toggle);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  test('labels the toggle for expanding when the charts are collapsed', async () => {
    await renderWithProviders(<EvalRunAssessmentHeaderToggle areChartsCollapsed onToggle={jest.fn()} />);

    expect(screen.getByRole('button', { name: 'Expand assessment charts' })).toHaveAttribute('aria-expanded', 'false');
  });
});
