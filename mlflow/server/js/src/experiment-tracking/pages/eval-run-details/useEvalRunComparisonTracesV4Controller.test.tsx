import { useEffect, useState, type ReactNode } from 'react';
import { beforeEach, describe, expect, it } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { SESSION_ID_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { setupServer } from '@databricks/web-shared/test-utils/setup-msw';
import { rest } from 'msw';
import { FilterOp } from '@databricks/web-shared/traces-table';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { useLocation } from '../../../common/utils/RoutingUtils';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '../../../common/utils/RoutingTestUtils';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { useEvalRunComparisonTracesV4Controller } from './useEvalRunComparisonTracesV4Controller';

const runUuidInFilter = (filter: string) => filter.match(/attributes\.run_id = '([^']*)'/)?.[1];

describe('useEvalRunComparisonTracesV4Controller', () => {
  const { server } = setupServer();
  const offPageId = '0123456789abcdef0123456789abcdef';
  let pageTraces: ModelTraceInfoV3[];
  let selectedTrace: ModelTraceInfoV3 | undefined;
  let searchReady: Promise<void>;
  let lastSearch = '';
  const LocationSpy = () => {
    const search = useLocation().search;
    useEffect(() => {
      lastSearch = search;
    }, [search]);
    return null;
  };

  beforeEach(() => {
    pageTraces = Array.from({ length: 26 }, (_, index) =>
      makeTrace(`trace-${index}`, {
        request_time: new Date(Date.UTC(2026, 0, 1, 0, 0, 30 - index)).toISOString(),
      }),
    );
    selectedTrace = undefined;
    searchReady = Promise.resolve();
    server.use(
      rest.post('*/ajax-api/2.0/mlflow/runs/search', (_req, res, ctx) => res(ctx.json({ runs: [] }))),
      rest.post('*/ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
        await searchReady;
        const body = await req.json<{ filter: string }>();
        if (!body.filter.includes('attributes.request_id')) {
          return res(ctx.json({ traces: pageTraces }));
        }
        // The exact-id lookup of a selected trace matches only the run's own, non-scorer traces.
        const runUuid = runUuidInFilter(body.filter);
        const matches =
          selectedTrace &&
          selectedTrace.trace_metadata?.['mlflow.sourceRun'] === runUuid &&
          !selectedTrace.tags?.['mlflow.trace.sourceScorer'];
        return res(ctx.json({ traces: matches ? [selectedTrace] : [] }));
      }),
      rest.get('*/ajax-api/3.0/mlflow/traces/batchGet', (_req, res, ctx) => res(ctx.json({ traces: [] }))),
    );
  });

  const renderController = async ({
    query = '',
    runUuids = ['baseline', 'candidate'],
    initialOutcomeFilter,
  }: {
    query?: string;
    runUuids?: string[];
    initialOutcomeFilter?: Parameters<typeof useEvalRunComparisonTracesV4Controller>[0]['initialOutcomeFilter'];
  } = {}) => {
    let currentRunUuids = runUuids;
    const Wrapper = ({ children }: { children?: ReactNode }) => {
      // An isolated client prevents comparison caches leaking across renders.
      const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }));
      return (
        <IntlProvider locale="en">
          <QueryClientProvider client={client}>
            <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
              <TestRouter
                routes={[
                  testRoute(
                    <>
                      <LocationSpy />
                      {children}
                    </>,
                    '/p',
                  ),
                ]}
                initialEntries={[`/p?compare=true&compareRuns=candidate${query}`]}
              />
            </MockedReduxStoreProvider>
          </QueryClientProvider>
        </IntlProvider>
      );
    };
    const hook = renderHook(
      () =>
        useEvalRunComparisonTracesV4Controller({
          experimentId: 'experiment',
          runUuids: currentRunUuids,
          initialOutcomeFilter,
        }),
      { wrapper: Wrapper },
    );
    await waitForRoutesToBeRendered();
    return {
      ...hook,
      setRunUuids: (next: string[]) => {
        currentRunUuids = next;
        hook.rerender();
      },
    };
  };

  const makeOutcomeTrace = (id: string, quality: [number, number], helpfulness: [number, number]): ModelTraceInfoV3 =>
    makeTrace(id, {
      assessments: [
        makeFeedbackAssessment('quality', quality[0], {
          metadata: { 'mlflow.assessment.sourceRunId': 'baseline' },
        }),
        makeFeedbackAssessment('quality', quality[1], {
          metadata: { 'mlflow.assessment.sourceRunId': 'candidate' },
        }),
        makeFeedbackAssessment('helpfulness', helpfulness[0], {
          metadata: { 'mlflow.assessment.sourceRunId': 'baseline' },
        }),
        makeFeedbackAssessment('helpfulness', helpfulness[1], {
          metadata: { 'mlflow.assessment.sourceRunId': 'candidate' },
        }),
      ],
    });

  it('filters all loaded groups before aggregation and pagination, and restores results on clear', async () => {
    pageTraces = Array.from({ length: 30 }, (_, index) =>
      makeTrace(`t-${index}`, { state: index < 27 ? 'ERROR' : 'OK' }),
    );
    const { result } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(30));
    act(() => result.current.setPageIndex(2));
    act(() => result.current.criteria.setFilterModel([{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]));
    expect(result.current.pageIndex).toBe(1);
    expect(result.current.totalRows).toBe(27);
    expect(result.current.rows).toHaveLength(25);
    expect(result.current.columnAggregates['state']?.values.map(({ displayValue }) => displayValue)).toEqual([
      '100%',
      '100%',
    ]);
    act(() => result.current.criteria.clearFilters());
    expect(result.current.totalRows).toBe(30);
    expect(result.current.columnAggregates['state']?.values.map(({ displayValue }) => displayValue)).toEqual([
      '90%',
      '90%',
    ]);
  });

  it('sorts complete groups before pagination and resets the current page', async () => {
    pageTraces = Array.from({ length: 26 }, (_, index) => makeTrace(`t-${index}`, { execution_duration: `${index}s` }));
    const { result } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(26));
    act(() => result.current.setPageIndex(2));
    act(() => result.current.criteria.setSort('duration', 'desc'));
    expect(result.current.pageIndex).toBe(1);
    expect(
      result.current.rows.map((row) => row.runs.map((run) => ('trace' in run ? run.trace?.trace_id : undefined))),
    ).toEqual(Array.from({ length: 25 }, (_, index) => [`t-${25 - index}`, `t-${25 - index}`]));
  });

  it('blocks results for unsupported persisted filters until explicitly cleared', async () => {
    const model = [{ field: 'span_name', operator: FilterOp.EQUALS, value: 'ignored' }];
    const { result } = await renderController({
      query: `&comparisonFilters=${encodeURIComponent(JSON.stringify(model))}`,
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.filterError).toBe('Complete or remove unsupported filters before applying.');
    expect(result.current.totalRows).toBe(0);
    act(() => result.current.criteria.clearFilters());
    expect(result.current.totalRows).toBe(26);
  });

  it('paginates matched inputs and deduplicates physical traces for page-local navigation', async () => {
    const { result } = await renderController();
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.totalRows).toBe(26);
      expect(result.current.rows).toHaveLength(25);
      expect(result.current.pageTraces.map((trace) => trace.trace_id)).toEqual(
        pageTraces.slice(0, 25).map((trace) => trace.trace_id),
      );
    });
    act(() => result.current.setPageIndex(2));
    expect(result.current.rows).toHaveLength(1);
    expect(result.current.pageTraces.map((trace) => trace.trace_id)).toEqual(['trace-25']);
    expect(result.current.hasNext).toBe(false);
    expect(result.current.hasPrev).toBe(true);
  });

  it('resets to page one after committed search while keeping matching-row siblings', async () => {
    const { result } = await renderController({ query: '&q=ignored&page=9&pageSize=100&groupBy=session' });
    await waitFor(() => expect(result.current.totalRows).toBe(26));
    expect(result.current.pageSize).toBe(25);
    act(() => result.current.setPageIndex(2));
    act(() => result.current.searchInput.setInput('trace-0'));
    act(() => result.current.searchInput.submit());
    expect(result.current.pageIndex).toBe(1);
    expect(
      result.current.rows.map((row) => row.runs.map((run) => ('trace' in run ? run.trace?.trace_id : undefined))),
    ).toEqual([['trace-0', 'trace-0']]);
    expect(new URLSearchParams(lastSearch).get('compareRuns')).toBe('candidate');
  });

  it('filters rows by baseline-relative outcomes and recomputes summaries before pagination', async () => {
    pageTraces = [
      makeOutcomeTrace('both-improve', [1, 2], [1, 2]),
      makeOutcomeTrace('quality-only', [1, 2], [2, 1]),
      makeOutcomeTrace('both-regress', [2, 1], [2, 1]),
    ];
    const { result } = await renderController();
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.totalRows).toBe(3);
      expect(result.current.assessmentInfos.map((info) => info.name)).toEqual(
        expect.arrayContaining(['quality', 'helpfulness']),
      );
    });
    const selections = () => result.current.outcomeFilters.map(({ id, ...filter }) => filter);

    const qualityImprovement = {
      columnId: 'assessment:quality',
      runUuid: 'candidate',
      outcome: 'improvement' as const,
    };
    const helpfulnessImprovement = {
      columnId: 'assessment:helpfulness',
      runUuid: 'candidate',
      outcome: 'improvement' as const,
    };
    const qualityRegression = { ...qualityImprovement, outcome: 'regression' as const };
    expect(result.current.aggregateMethods['assessment:quality'] ?? 'mean').toBe('mean');
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '1.33', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '1.67',
          displayDelta: '+0.33',
          delta: 0.33333333333333326,
          percentDelta: 25,
          changeCounts: { improved: 2, regressed: 1 },
        },
      ],
    });
    expect(result.current.filterOptionAggregates['assessment:quality']).toEqual(
      result.current.columnAggregates['assessment:quality'],
    );

    act(() => result.current.toggleOutcomeFilter(qualityImprovement));
    expect(selections()).toEqual([qualityImprovement]);
    expect(result.current.totalRows).toBe(2);
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '1', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '2',
          displayDelta: '+1',
          delta: 1,
          percentDelta: 100,
          changeCounts: { improved: 2, regressed: 0 },
        },
      ],
    });
    expect(result.current.filterOptionAggregates['assessment:quality']?.values[1].changeCounts).toEqual({
      improved: 2,
      regressed: 1,
    });

    act(() => result.current.toggleOutcomeFilter(qualityRegression));
    expect(selections()).toEqual([qualityRegression]);
    expect(result.current.totalRows).toBe(1);
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '2', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '1',
          displayDelta: '-1',
          delta: -1,
          percentDelta: -50,
          changeCounts: { improved: 0, regressed: 1 },
        },
      ],
    });

    act(() => result.current.toggleOutcomeFilter(qualityRegression));
    expect(result.current.outcomeFilters).toEqual([]);
    expect(result.current.totalRows).toBe(3);

    act(() => result.current.toggleOutcomeFilter(qualityImprovement));
    act(() => result.current.toggleOutcomeFilter(helpfulnessImprovement));
    expect(result.current.totalRows).toBe(1);
    expect(result.current.rows[0].runs[0]).toMatchObject({ trace: { trace_id: 'both-improve' } });
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '1', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '2',
          displayDelta: '+1',
          delta: 1,
          percentDelta: 100,
          changeCounts: { improved: 1, regressed: 0 },
        },
      ],
    });
    expect(result.current.columnAggregates['assessment:helpfulness']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '1', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '2',
          displayDelta: '+1',
          delta: 1,
          percentDelta: 100,
          changeCounts: { improved: 1, regressed: 0 },
        },
      ],
    });

    const originalFilter = result.current.outcomeFilters[0];
    act(() => result.current.replaceOutcomeFilter(originalFilter, { ...originalFilter, outcome: 'regression' }));
    expect(selections()).toEqual([qualityRegression, helpfulnessImprovement]);
    expect(result.current.outcomeFilters[0].id).toBe(originalFilter.id);
    expect(result.current.totalRows).toBe(0);
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '—', displayDelta: undefined, changeCounts: undefined },
        { displayValue: '—', displayDelta: undefined, changeCounts: undefined },
      ],
    });
    expect(result.current.columnAggregates['assessment:helpfulness']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '—', displayDelta: undefined, changeCounts: undefined },
        { displayValue: '—', displayDelta: undefined, changeCounts: undefined },
      ],
    });
    expect(result.current.filterOptionAggregates['assessment:quality']?.values[1].changeCounts).toEqual({
      improved: 2,
      regressed: 1,
    });

    act(() => result.current.removeOutcomeFilter(result.current.outcomeFilters[1]));
    expect(result.current.totalRows).toBe(1);
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '2', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '1',
          displayDelta: '-1',
          delta: -1,
          percentDelta: -50,
          changeCounts: { improved: 0, regressed: 1 },
        },
      ],
    });
    act(() => result.current.clearOutcomeFilters());
    expect(result.current.outcomeFilters).toEqual([]);
    expect(result.current.totalRows).toBe(3);
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: 'AVG',
      values: [
        { displayValue: '1.33', displayDelta: undefined, changeCounts: undefined },
        {
          displayValue: '1.67',
          displayDelta: '+0.33',
          delta: 0.33333333333333326,
          percentDelta: 25,
          changeCounts: { improved: 2, regressed: 1 },
        },
      ],
    });

    act(() => result.current.setAggregateMethod('assessment:quality', 'median'));
    expect(result.current.aggregateMethods['assessment:quality']).toBe('median');
    expect(result.current.columnAggregates['assessment:quality'].label).toBe('MEDIAN');
  });

  it('replaces outcome filters atomically when the edited badge conflicts with another filter key', async () => {
    pageTraces = [makeOutcomeTrace('both-improve', [1, 2], [1, 2]), makeOutcomeTrace('both-regress', [2, 1], [2, 1])];
    const { result } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(2));
    const selections = () => result.current.outcomeFilters.map(({ id, ...filter }) => filter);

    const qualityImprovement = {
      columnId: 'assessment:quality',
      runUuid: 'candidate',
      outcome: 'improvement' as const,
    };
    const helpfulnessImprovement = {
      columnId: 'assessment:helpfulness',
      runUuid: 'candidate',
      outcome: 'improvement' as const,
    };
    const helpfulnessRegression = { ...helpfulnessImprovement, outcome: 'regression' as const };

    act(() => result.current.toggleOutcomeFilter(qualityImprovement));
    act(() => result.current.toggleOutcomeFilter(helpfulnessImprovement));
    expect(selections()).toEqual([qualityImprovement, helpfulnessImprovement]);

    const originalFilter = result.current.outcomeFilters[0];
    act(() => result.current.replaceOutcomeFilter(originalFilter, { ...originalFilter, ...helpfulnessRegression }));
    expect(selections()).toEqual([helpfulnessRegression]);
    expect(result.current.outcomeFilters[0].id).toBe(originalFilter.id);
    expect(result.current.totalRows).toBe(1);
    expect(result.current.rows[0].runs[0]).toMatchObject({ trace: { trace_id: 'both-regress' } });
  });

  it('removes outcome filters for runs no longer in the comparison', async () => {
    pageTraces = [makeOutcomeTrace('both-improve', [1, 2], [1, 2])];
    const { result, setRunUuids } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(1));

    act(() =>
      result.current.toggleOutcomeFilter({
        columnId: 'assessment:quality',
        runUuid: 'candidate',
        outcome: 'improvement',
      }),
    );
    expect(result.current.outcomeFilters).toHaveLength(1);

    act(() => setRunUuids(['baseline', 'replacement']));
    await waitFor(() => expect(result.current.outcomeFilters).toEqual([]));
  });

  it('removes assessment outcome filters when search leaves no filterable rows', async () => {
    pageTraces = [makeOutcomeTrace('both-improve', [1, 2], [1, 2])];
    const { result } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(1));

    act(() =>
      result.current.toggleOutcomeFilter({
        columnId: 'assessment:quality',
        runUuid: 'candidate',
        outcome: 'improvement',
      }),
    );
    act(() => result.current.criteria.setSearch('no matching trace'));
    await waitFor(() => expect(result.current.totalRows).toBe(0));
    await waitFor(() => expect(result.current.outcomeFilters).toEqual([]));
  });

  it('uses search-matched assessment metadata for outcome classification and filter options', async () => {
    pageTraces = [
      makeTrace('retained', {
        assessments: [
          makeFeedbackAssessment('quality', 1, {
            metadata: { 'mlflow.assessment.sourceRunId': 'baseline' },
          }),
          makeFeedbackAssessment('quality', 2, {
            metadata: { 'mlflow.assessment.sourceRunId': 'candidate' },
          }),
        ],
      }),
      makeTrace('excluded', {
        assessments: [
          makeFeedbackAssessment('quality', 'other', {
            metadata: { 'mlflow.assessment.sourceRunId': 'baseline' },
          }),
          makeFeedbackAssessment('quality', 'other', {
            metadata: { 'mlflow.assessment.sourceRunId': 'candidate' },
          }),
        ],
      }),
    ];
    const { result } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(2));

    act(() => result.current.searchInput.setInput('retained'));
    act(() => result.current.searchInput.submit());

    expect(result.current.assessmentInfos).toEqual(
      expect.arrayContaining([expect.objectContaining({ name: 'quality', dtype: 'numeric' })]),
    );
    expect(result.current.assessmentInfos).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ dtype: 'string' })]),
    );

    const qualityImprovement = {
      columnId: 'assessment:quality',
      runUuid: 'candidate',
      outcome: 'improvement' as const,
    };
    act(() => result.current.toggleOutcomeFilter(qualityImprovement));
    expect(result.current.totalRows).toBe(1);
    expect(result.current.rows[0].runs[0]).toMatchObject({ trace: { trace_id: 'retained' } });

    act(() => result.current.searchInput.clear());
    await waitFor(() => expect(result.current.outcomeFilters).toEqual([]));
    expect(result.current.totalRows).toBe(2);
  });

  it('resets pagination when an outcome filter narrows the search-retained rows', async () => {
    pageTraces = Array.from({ length: 26 }, (_, index) =>
      makeOutcomeTrace(`outcome-${index}`, index === 25 ? [1, 2] : [1, 1], [1, 1]),
    );
    const { result } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(26));
    act(() => result.current.setPageIndex(2));
    expect(result.current.pageIndex).toBe(2);

    act(() =>
      result.current.toggleOutcomeFilter({
        columnId: 'assessment:quality',
        runUuid: 'candidate',
        outcome: 'improvement',
      }),
    );
    expect(result.current.pageIndex).toBe(1);
    expect(result.current.totalRows).toBe(1);
    expect(result.current.rows[0].runs[0]).toMatchObject({ trace: { trace_id: 'outcome-25' } });
  });

  // ui-test level: unit; the controller exposes summary types, values and search transitions without rendering a table.
  it.each([
    {
      type: 'numeric',
      baseline: 2,
      candidate: 6,
      label: 'AVG',
      baselineScore: '2',
      candidateScore: '6',
      delta: '+4',
      aggregateDelta: 4,
      percentDelta: 200,
    },
    {
      type: 'boolean',
      baseline: false,
      candidate: true,
      label: 'TRUE',
      baselineScore: '0%',
      candidateScore: '100%',
      delta: '+100',
      aggregateDelta: 1,
      percentDelta: undefined,
    },
    {
      type: 'pass-fail',
      baseline: 'no',
      candidate: 'yes',
      label: 'PASS',
      baselineScore: '0%',
      candidateScore: '100%',
      delta: '+100',
      aggregateDelta: 1,
      percentDelta: undefined,
    },
  ])(
    'infers $type summaries from retained groups and restores categorical summaries when search clears',
    async ({ baseline, candidate, label, baselineScore, candidateScore, delta, aggregateDelta, percentDelta }) => {
      pageTraces = [
        makeTrace('retained', {
          assessments: [
            makeFeedbackAssessment('quality', baseline, { metadata: { 'mlflow.assessment.sourceRunId': 'baseline' } }),
            makeFeedbackAssessment('quality', candidate, {
              metadata: { 'mlflow.assessment.sourceRunId': 'candidate' },
            }),
          ],
        }),
        makeTrace('excluded', { assessments: [makeFeedbackAssessment('quality', 'other')] }),
      ];
      const { result } = await renderController();
      await waitFor(() =>
        expect(result.current.columnAggregates['assessment:quality']?.values).toEqual([
          { displayValue: '2 values' },
          { displayValue: '2 values' },
        ]),
      );

      act(() => result.current.searchInput.setInput('retained'));
      act(() => result.current.searchInput.submit());

      expect(result.current.columnAggregates['assessment:quality']).toEqual({
        label,
        values: [
          { displayValue: baselineScore, displayDelta: undefined, changeCounts: undefined },
          {
            displayValue: candidateScore,
            displayDelta: delta,
            delta: aggregateDelta,
            percentDelta,
            changeCounts: { improved: 1, regressed: 0 },
          },
        ],
      });

      act(() => result.current.searchInput.clear());
      expect(result.current.columnAggregates['assessment:quality']).toEqual({
        label: undefined,
        values: [{ displayValue: '2 values' }, { displayValue: '2 values' }],
      });
    },
  );

  // ui-test level: unit; returned summaries must include off-page, run-scoped copies of retained traces.
  it('includes off-page assessment types and trace failures in summaries across pagination', async () => {
    pageTraces = Array.from({ length: 26 }, (_, index) =>
      makeTrace(`retained-${index}`, {
        request_time: new Date(Date.UTC(2026, 0, 1, 0, 0, 30 - index)).toISOString(),
        state: index === 25 ? 'ERROR' : 'OK',
        assessments: [
          makeFeedbackAssessment('quality', 2, { metadata: { 'mlflow.assessment.sourceRunId': 'baseline' } }),
          makeFeedbackAssessment('quality', index === 25 ? 'other' : 6, {
            metadata: { 'mlflow.assessment.sourceRunId': 'candidate' },
          }),
        ],
      }),
    );
    const { result } = await renderController();
    await waitFor(() => expect(result.current.totalRows).toBe(26));
    act(() => result.current.searchInput.setInput('retained'));
    act(() => result.current.searchInput.submit());

    expect(result.current.hasNext).toBe(true);
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: undefined,
      values: [{ displayValue: '1 value' }, { displayValue: '2 values' }],
    });
    expect(result.current.columnAggregates['state']).toEqual({
      label: 'ERROR RATE',
      values: [
        { displayValue: '4%', displayDelta: undefined, changeCounts: undefined },
        { displayValue: '4%', displayDelta: undefined, delta: 0, percentDelta: 0, changeCounts: undefined },
      ],
    });

    act(() => result.current.setPageIndex(2));
    expect(result.current.pageTraces.map((trace) => trace.trace_id)).toEqual(['retained-25']);
    expect(result.current.columnAggregates['assessment:quality']).toEqual({
      label: undefined,
      values: [{ displayValue: '1 value' }, { displayValue: '2 values' }],
    });
    expect(result.current.columnAggregates['state']).toEqual({
      label: 'ERROR RATE',
      values: [
        { displayValue: '4%', displayDelta: undefined, changeCounts: undefined },
        { displayValue: '4%', displayDelta: undefined, delta: 0, percentDelta: 0, changeCounts: undefined },
      ],
    });
  });

  // ui-test level: unit; empty search results clear summaries without changing column availability.
  it('clears assessment summaries but keeps columns when search matches no groups', async () => {
    pageTraces = [makeTrace('retained', { assessments: [makeFeedbackAssessment('quality', 2)] })];
    const { result } = await renderController();
    await waitFor(() => expect(result.current.columnAggregates['assessment:quality']?.label).toBe('AVG'));
    const columnIds = result.current.assessments.columnDefs.map((column) => column.id);

    act(() => result.current.searchInput.setInput('missing'));
    act(() => result.current.searchInput.submit());

    expect(result.current.rows).toEqual([]);
    expect(result.current.columnAggregates['assessment:quality']).toBeUndefined();
    expect(result.current.columnAggregates['duration']).toEqual({
      label: 'AVG',
      values: [{ displayValue: '—' }, { displayValue: '—' }],
    });
    expect(result.current.assessments.columnDefs.map((column) => column.id)).toEqual(columnIds);
  });

  it('opens an off-page member using its canonical trace identifier', async () => {
    selectedTrace = makeTrace(offPageId, { trace_metadata: { 'mlflow.sourceRun': 'candidate' } });
    const { result } = await renderController({ query: `&selectedEvaluationId=${offPageId}&spanId=span-1` });
    await waitFor(() => expect(result.current.activeTraceId).toBe(`trace:/cat.sch/${offPageId}`));
    expect(result.current.pageTraces.some((trace) => trace.trace_id === offPageId)).toBe(false);
    expect(new URLSearchParams(lastSearch).get('compare')).toBe('true');
    expect(new URLSearchParams(lastSearch).get('compareRuns')).toBe('candidate');
  });

  it.each(['nonmember', 'foreign-location', 'scorer'])(
    'clears a %s deeplink without losing the comparison selection',
    async (kind) => {
      selectedTrace = makeTrace(offPageId, {
        trace_metadata: { 'mlflow.sourceRun': kind === 'scorer' ? 'candidate' : 'other-run' },
        tags: kind === 'scorer' ? { 'mlflow.trace.sourceScorer': 'judge' } : {},
      });
      const id = kind === 'foreign-location' ? `trace:/other.schema/${offPageId}` : offPageId;
      const { result } = await renderController({
        query: `&selectedEvaluationId=${encodeURIComponent(id)}&spanId=span-1`,
      });
      await waitFor(() => expect(new URLSearchParams(lastSearch).has('selectedEvaluationId')).toBe(false));
      expect(result.current.activeTraceId).toBeUndefined();
      expect(new URLSearchParams(lastSearch).get('compareRuns')).toBe('candidate');
      expect(new URLSearchParams(lastSearch).get('compare')).toBe('true');
      expect(new URLSearchParams(lastSearch).has('spanId')).toBe(false);
    },
  );

  it('keeps a selected trace available while the comparison refetches', async () => {
    pageTraces = [makeTrace(offPageId, { trace_metadata: { 'mlflow.sourceRun': 'baseline' } })];
    const { result } = await renderController({ query: `&selectedEvaluationId=${offPageId}` });
    await waitFor(() => {
      expect(result.current.activeTraceId).toBe(`trace:/cat.sch/${offPageId}`);
      expect(result.current.isFetching).toBe(false);
    });
    let release = () => {};
    searchReady = new Promise<void>((resolve) => {
      release = resolve;
    });
    try {
      act(() => {
        void result.current.refetch();
      });
      await waitFor(() => expect(result.current.isFetching).toBe(true));
      expect(result.current.isLoading).toBe(false);
      expect(result.current.activeTraceId).toBe(`trace:/cat.sch/${offPageId}`);
    } finally {
      release();
    }
    await waitFor(() => expect(result.current.isFetching).toBe(false));
    expect(result.current.activeTraceId).toBe(`trace:/cat.sch/${offPageId}`);
  });

  it('waits for the run searches before opening a selected trace', async () => {
    pageTraces = [makeTrace(offPageId, { trace_metadata: { 'mlflow.sourceRun': 'baseline' } })];
    let release = () => {};
    searchReady = new Promise<void>((resolve) => {
      release = resolve;
    });
    const { result } = await renderController({ query: `&selectedEvaluationId=${offPageId}` });
    try {
      await waitFor(() => expect(result.current.isLoading).toBe(true));
      expect(result.current.activeTraceId).toBeUndefined();
    } finally {
      act(() => release());
    }
    await waitFor(() => expect(result.current.activeTraceId).toBe(`trace:/cat.sch/${offPageId}`));
    expect(new URLSearchParams(lastSearch).get('selectedEvaluationId')).toBe(offPageId);
  });

  it('retains complete matched sessions only when one turn satisfies every trace filter and search', async () => {
    const sessionTrace = (id: string, sessionId: string, state: ModelTraceInfoV3['state'], input: string) =>
      makeTrace(id, { state, request_preview: input, trace_metadata: { [SESSION_ID_METADATA_KEY]: sessionId } });
    const baselineTraces = [
      sessionTrace('split-error', 'split-session', 'ERROR', 'unrelated input'),
      sessionTrace('split-search', 'split-session', 'OK', 'needle'),
      sessionTrace('matched-error', 'matched-session', 'ERROR', 'needle'),
      sessionTrace('matched-other-turn', 'matched-session', 'OK', 'keep the entire conversation'),
    ];
    const candidateTraces = [
      sessionTrace('candidate-split', 'split-session', 'OK', 'different conversation'),
      sessionTrace('candidate-matched', 'matched-session', 'OK', 'different matched-run input'),
    ];
    server.use(
      rest.post('*/ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
        const body = await req.json<{ filter: string }>();
        return res(
          ctx.json({ traces: runUuidInFilter(body.filter) === 'baseline' ? baselineTraces : candidateTraces }),
        );
      }),
    );
    const { result } = await renderController({ query: '&comparisonGroupBy=session' });
    await waitFor(() => expect(result.current.totalRows).toBe(2));

    act(() => result.current.criteria.setFilterModel([{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]));
    act(() => result.current.criteria.setSearch('needle'));

    expect(result.current.totalRows).toBe(1);
    expect(
      result.current.rows.map((row) =>
        row.kind === 'session'
          ? row.runs.map((run) => ({
              session: run.sessionId,
              traces: run.traces.map((trace) => trace.trace_id),
            }))
          : undefined,
      ),
    ).toEqual([
      [
        { session: 'matched-session', traces: ['matched-error', 'matched-other-turn'] },
        { session: 'matched-session', traces: ['candidate-matched'] },
      ],
    ]);
    expect(result.current.columnAggregates['state']?.values.map((value) => value.displayValue)).toEqual(['0%', '0%']);
  });

  it('aggregates all loaded session summaries before paging without expansion changing rows or totals', async () => {
    pageTraces = Array.from({ length: 26 }, (_, index) =>
      [0, 1].map((turn) =>
        makeTrace(`session-${index}-turn-${turn}`, {
          request_time: new Date(Date.UTC(2026, 0, 1, index, turn)).toISOString(),
          trace_metadata: { [SESSION_ID_METADATA_KEY]: `session-${index}` },
          assessments: [makeFeedbackAssessment('quality', index === 25 ? 10 : turn * 2)],
        }),
      ),
    ).flat();
    const { result } = await renderController({ query: '&comparisonGroupBy=session' });
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.totalRows).toBe(26);
      expect(result.current.rows).toHaveLength(25);
    });
    expect(result.current.rows).toHaveLength(25);
    expect(result.current.pageTraces).toHaveLength(50);
    expect(result.current.expandedComparisonGroups.size).toBe(26);
    expect(result.current.expandedComparisonRuns).toEqual(new Map());
    expect(result.current.columnAggregates['assessment:quality']?.values.map((value) => value.displayValue)).toEqual([
      '1.35',
      '1.35',
    ]);

    const groupId = result.current.rows[0].id;
    const loadedTraceIds = result.current.pageTraces.map((trace) => trace.trace_id);
    act(() => result.current.toggleComparisonGroup(groupId));
    act(() => result.current.toggleComparisonRun(groupId, 'baseline'));
    act(() => result.current.toggleComparisonRun(groupId, 'candidate'));
    expect(result.current.rows).toHaveLength(25);
    expect(result.current.pageTraces.map((trace) => trace.trace_id)).toEqual(loadedTraceIds);
    expect(result.current.pageIndex).toBe(1);
    act(() => result.current.setPageIndex(2));

    expect(result.current.rows).toHaveLength(1);
    expect(result.current.expandedComparisonGroups.has(result.current.rows[0].id)).toBe(true);
    expect(result.current.pageTraces).toHaveLength(2);
    expect(result.current.totalRows).toBe(26);
    expect(result.current.columnAggregates['assessment:quality']?.values.map((value) => value.displayValue)).toEqual([
      '1.35',
      '1.35',
    ]);
  });

  it('keeps group and run expansion independent while preserving sibling selections', async () => {
    pageTraces = ['first', 'second'].map((sessionId) =>
      makeTrace(`${sessionId}-turn`, { trace_metadata: { [SESSION_ID_METADATA_KEY]: sessionId } }),
    );
    const { result } = await renderController({ query: '&comparisonGroupBy=session' });
    await waitFor(() => expect(result.current.totalRows).toBe(2));
    const [firstGroup, secondGroup] = result.current.rows.map((row) => row.id);
    expect(result.current.expandedComparisonGroups).toEqual(new Set([firstGroup, secondGroup]));
    expect(result.current.expandedComparisonRuns).toEqual(new Map());

    act(() => {
      result.current.toggleComparisonRun(firstGroup, 'baseline');
      result.current.toggleComparisonRun(firstGroup, 'candidate');
      result.current.toggleComparisonRun(secondGroup, 'candidate');
    });
    expect(result.current.expandedComparisonGroups).toEqual(new Set([firstGroup, secondGroup]));
    expect(result.current.expandedComparisonRuns).toEqual(
      new Map([
        [firstGroup, new Set(['baseline', 'candidate'])],
        [secondGroup, new Set(['candidate'])],
      ]),
    );

    act(() => result.current.toggleComparisonGroup(firstGroup));
    expect(result.current.expandedComparisonGroups).toEqual(new Set([secondGroup]));
    expect(result.current.expandedComparisonRuns).toEqual(
      new Map([
        [firstGroup, new Set(['baseline', 'candidate'])],
        [secondGroup, new Set(['candidate'])],
      ]),
    );
    act(() => result.current.toggleComparisonRun(firstGroup, 'baseline'));
    expect(result.current.expandedComparisonRuns).toEqual(
      new Map([
        [firstGroup, new Set(['candidate'])],
        [secondGroup, new Set(['candidate'])],
      ]),
    );
    act(() => result.current.toggleComparisonRun(firstGroup, 'candidate'));
    expect(result.current.expandedComparisonRuns.get(firstGroup)?.size ?? 0).toBe(0);
    expect(result.current.expandedComparisonRuns.get(secondGroup)).toEqual(new Set(['candidate']));
    act(() => result.current.toggleComparisonGroup(firstGroup));
    expect(result.current.expandedComparisonGroups).toEqual(new Set([firstGroup, secondGroup]));
  });

  it.each(['pagination', 'filtering', 'sorting', 'search'])(
    'preserves collapsed session and expanded run identities through %s',
    async (change) => {
      pageTraces = Array.from({ length: 26 }, (_, index) =>
        makeTrace(`session-${index}-turn`, {
          request_time: new Date(Date.UTC(2026, 0, 1, 0, index)).toISOString(),
          state: index === 25 ? 'ERROR' : 'OK',
          trace_metadata: { [SESSION_ID_METADATA_KEY]: `session-${index}` },
        }),
      );
      const { result } = await renderController({ query: '&comparisonGroupBy=session' });
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
        expect(result.current.totalRows).toBe(26);
        expect(
          result.current.rows.some(
            (row) => row.kind === 'session' && row.runs.some((run) => run.sessionId === 'session-25'),
          ),
        ).toBe(true);
      });
      const group = result.current.rows.find(
        (row) => row.kind === 'session' && row.runs.some((run) => run.sessionId === 'session-25'),
      );
      if (!group) {
        throw new Error('Expected newest session on the first page');
      }
      const otherGroups = new Set(result.current.expandedComparisonGroups);
      otherGroups.delete(group.id);
      act(() => {
        result.current.toggleComparisonGroup(group.id);
        result.current.toggleComparisonRun(group.id, 'candidate');
      });
      if (change === 'pagination') {
        act(() => result.current.setPageIndex(2));
      } else if (change === 'filtering') {
        act(() => result.current.criteria.setFilterModel([{ field: 'state', operator: FilterOp.EQUALS, value: 'OK' }]));
      } else if (change === 'sorting') {
        act(() => result.current.criteria.setSort('start_time', 'asc'));
      } else {
        act(() => result.current.criteria.setSearch('session-0-turn'));
      }
      expect(result.current.rows.map((row) => row.id)).not.toContain(group.id);
      expect(result.current.expandedComparisonGroups).toEqual(otherGroups);
      expect(result.current.expandedComparisonRuns).toEqual(new Map([[group.id, new Set(['candidate'])]]));

      if (change === 'pagination') {
        act(() => result.current.setPageIndex(1));
      } else if (change === 'filtering') {
        act(() => result.current.criteria.clearFilters());
      } else if (change === 'sorting') {
        act(() => result.current.criteria.setSort('start_time', 'desc'));
      } else {
        act(() => result.current.criteria.setSearch(''));
      }
      expect(result.current.rows.map((row) => row.id)).toContain(group.id);
      expect(result.current.expandedComparisonGroups).toEqual(otherGroups);
      expect(result.current.expandedComparisonRuns).toEqual(new Map([[group.id, new Set(['candidate'])]]));
    },
  );

  it('opens newly loaded sessions after refresh while retaining collapsed parents and expanded runs', async () => {
    pageTraces = ['first', 'second'].map((sessionId) =>
      makeTrace(`${sessionId}-turn`, { trace_metadata: { [SESSION_ID_METADATA_KEY]: sessionId } }),
    );
    const { result } = await renderController({ query: '&comparisonGroupBy=session' });
    await waitFor(() => expect(result.current.totalRows).toBe(2));
    const collapsedGroup = result.current.rows.find(
      (row) => row.kind === 'session' && row.runs.some((run) => run.sessionId === 'first'),
    );
    if (!collapsedGroup) {
      throw new Error('Expected first session fixture');
    }
    const initialGroupIds = result.current.rows.map((row) => row.id);
    act(() => {
      result.current.toggleComparisonGroup(collapsedGroup.id);
      result.current.toggleComparisonRun(collapsedGroup.id, 'candidate');
    });
    pageTraces = [
      ...pageTraces,
      makeTrace('first-added-turn', { trace_metadata: { [SESSION_ID_METADATA_KEY]: 'first' } }),
      makeTrace('new-session-turn', { trace_metadata: { [SESSION_ID_METADATA_KEY]: 'new-session' } }),
    ];

    await act(async () => {
      await result.current.refetch();
    });
    await waitFor(() => expect(result.current.totalRows).toBe(3));

    const newGroup = result.current.rows.find(
      (row) => row.kind === 'session' && row.runs.some((run) => run.sessionId === 'new-session'),
    );
    if (!newGroup) {
      throw new Error('Expected refreshed session fixture');
    }
    expect(result.current.pageTraces.map((trace) => trace.trace_id).sort()).toEqual([
      'first-added-turn',
      'first-turn',
      'new-session-turn',
      'second-turn',
    ]);
    expect(result.current.expandedComparisonGroups).toEqual(
      new Set([...initialGroupIds.filter((id) => id !== collapsedGroup.id), newGroup.id]),
    );
    expect(result.current.expandedComparisonRuns).toEqual(new Map([[collapsedGroup.id, new Set(['candidate'])]]));
  });

  it('resets pagination and isolates column preferences when switching to sessions', async () => {
    pageTraces = Array.from({ length: 30 }, (_, index) =>
      makeTrace(`turn-${index}`, {
        trace_metadata: { [SESSION_ID_METADATA_KEY]: `session-${Math.floor(index / 2)}` },
      }),
    );
    const { result } = await renderController({ query: '&comparisonSort=duration&comparisonDir=asc' });
    await waitFor(() => expect(result.current.totalRows).toBe(30));
    expect(result.current.persistenceId).toBe('experiment.run-comparison');
    act(() => result.current.setPageIndex(2));
    act(() => result.current.criteria.setIsGroupedBySession(true));
    expect(result.current.pageIndex).toBe(1);
    expect(result.current.totalRows).toBe(15);
    expect(result.current.persistenceId).toBe('experiment.run-comparison.sessions');
    expect(result.current.criteria.sort).toBe('duration');
    expect(result.current.criteria.dir).toBe('asc');
    expect(new URLSearchParams(lastSearch).get('compareRuns')).toBe('candidate');
  });

  it('leaves session grouping when opened with an outcome filter', async () => {
    pageTraces = [makeOutcomeTrace('improved', [1, 2], [1, 2]), makeOutcomeTrace('regressed', [2, 1], [2, 1])];
    const { result } = await renderController({
      query: '&comparisonGroupBy=session',
      initialOutcomeFilter: { columnId: 'assessment:quality', runUuid: 'candidate', outcome: 'improvement' },
    });
    await waitFor(() => expect(result.current.criteria.isGroupedBySession).toBe(false));
    await waitFor(() => expect(result.current.totalRows).toBe(1));
    await waitFor(() => expect(result.current.rows[0]?.runs[0]).toMatchObject({ trace: { trace_id: 'improved' } }));
    expect(new URLSearchParams(lastSearch).has('comparisonGroupBy')).toBe(false);
  });
});
