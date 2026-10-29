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
      makeFeedbackAssessment('tone', 'curt'),
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
  { name: 'tone', type: 'categorical' },
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
  currentScores,
  baselineScores,
  baselineRunHref,
}: {
  assessmentFilters?: AssessmentFilter[];
  toggleAssessmentFilter?: (name: string, value: unknown) => void;
  areChartsCollapsed?: boolean;
  currentScores?: Record<string, number>;
  baselineScores?: Record<string, number>;
  baselineRunHref?: string;
} = {}) =>
  buildEvalRunAssessmentColumnDefs({
    columnDefs: baseColumns,
    scorersByName: summarize(TRACES),
    assessmentFilters,
    toggleAssessmentFilter,
    runUuid: RUN_UUID,
    areChartsCollapsed,
    currentScores,
    baselineScores,
    baselineRunHref,
  });

const header = (name: RegExp) => screen.getByRole('columnheader', { name });
const pillIn = (name: RegExp) => within(header(name)).getByTestId('assessment-baseline-delta-pill');
const BASELINE_HREF = '/experiments/exp-1/evaluation-runs/baseline-1/details';
const SAVED_SCORES = { correctness: 0.84, relevance: 3.5, tone: 0.5 };

describe('evaluation run assessment column headers', () => {
  test('gives summarized assessment columns a rich summary header and leaves the rest as they are', () => {
    const columns = build();
    const byId = new Map(columns.map((column) => [column.id, column]));

    expect(byId.get('assessment:correctness')?.headerVariant).toBe('rich');
    expect(byId.get('assessment:relevance')?.headerVariant).toBe('rich');
    expect(byId.get('assessment:not_summarized')).toBe(baseColumns[4]);
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

  test("shows the run's saved metrics and their change from the baseline run", async () => {
    await renderWithProviders(
      <TracesTable
        {...tableProps(
          build({
            currentScores: SAVED_SCORES,
            baselineScores: { correctness: 0.8, relevance: 3.25, tone: 0.25 },
            baselineRunHref: BASELINE_HREF,
          }),
        )}
      />,
    );

    // A string scorer keeps its count of values, even with a saved metric.
    const tone = header(/^tone/);
    expect(tone).toHaveAccessibleName('tone STRING 2 values');
    expect(within(tone).queryByTestId('assessment-baseline-delta-pill')).not.toBeInTheDocument();
    // Rates change in percentage points; the summaries computed from the traces would say 75% and 3.
    expect(header(/^Correctness/)).toHaveAccessibleName(/^Correctness PASS 84% \+4 /);
    expect(pillIn(/^Correctness/)).toHaveAttribute('data-baseline-delta-direction', 'improvement');
    expect(header(/^relevance/)).toHaveAccessibleName('relevance AVG 3.5 +0.25');
    // A scorer without a saved metric keeps its summary and has no pill.
    expect(header(/^is_polite/)).toHaveAccessibleName(/^is_polite TRUE 67% /);
    expect(within(header(/^is_polite/)).queryByTestId('assessment-baseline-delta-pill')).not.toBeInTheDocument();
  });

  test('marks a regression, and an equal score as neutral', async () => {
    await renderWithProviders(
      <TracesTable
        {...tableProps(
          build({
            currentScores: SAVED_SCORES,
            baselineScores: { correctness: 0.9, relevance: 3.5 },
            baselineRunHref: BASELINE_HREF,
          }),
        )}
      />,
    );

    expect(pillIn(/^Correctness/)).toHaveTextContent('-6');
    expect(pillIn(/^Correctness/)).toHaveAttribute('data-baseline-delta-direction', 'regression');
    expect(pillIn(/^relevance/)).toHaveTextContent('+0.0');
    expect(pillIn(/^relevance/)).toHaveAttribute('data-baseline-delta-direction', 'neutral');
  });

  test('links to the baseline run and says when it has no score for a scorer', async () => {
    await renderWithProviders(
      <TracesTable
        {...tableProps(build({ currentScores: SAVED_SCORES, baselineScores: {}, baselineRunHref: BASELINE_HREF }))}
      />,
    );

    const pill = pillIn(/^Correctness/);
    expect(pill).toHaveTextContent('-');
    expect(pill).toHaveAttribute('data-baseline-delta-direction', 'neutral');
    await userEvent.hover(pill);

    expect(await screen.findByText(/does not have this scorer/)).toBeInTheDocument();
    const link = screen.getByRole('link', { name: 'Open evaluation baseline run in a new tab' });
    expect(link).toHaveAttribute('href', BASELINE_HREF);
    expect(link).toHaveAttribute('target', '_blank');
  });

  test('shows no change pills without a baseline run', async () => {
    await renderWithProviders(<TracesTable {...tableProps(build({ currentScores: SAVED_SCORES }))} />);

    expect(header(/^Correctness/)).toHaveAccessibleName(/^Correctness PASS 84% /);
    expect(screen.queryByTestId('assessment-baseline-delta-pill')).not.toBeInTheDocument();
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
