import { describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import type { TracesTableProps } from '@databricks/web-shared/traces-table/TracesTable';
import { makeTraces } from '../test-utils/mockTraces';
import { renderPage, state, env } from '../test-utils/tracesV4PageContentTestBed';

// Keep the real controller, requests, empty states, and pagination controls. Detailed row rendering
// is covered by TracesTable.test.tsx and TracesV4PageContent.test.tsx.
jest.mock('@databricks/web-shared/traces-table/TracesTable', () => ({
  TracesTable: ({ traces, isLoading }: TracesTableProps) =>
    isLoading ? null : (
      <div>
        {traces.map((trace) => (
          <span key={trace.trace_id}>{trace.trace_id}</span>
        ))}
      </div>
    ),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => ({ data: { tags: [] }, refetch: () => Promise.resolve({}) }),
}));

const findTraceRow = (traceId: string) => screen.findByText(traceId);

describe('TracesV4PageContent pagination', () => {
  test('Next sends the second page token and shows the second page; Prev does not refetch', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    // Page 1 must be a *full* page (rows === pageSize) for Next to enable — a short page is treated as
    // the last page regardless of the token. Default pageSize is 25, so return 25 rows.
    state.pages = {
      '': { traces: makeTraces(25, 'p1'), next_page_token: 'token-2' },
      'token-2': { traces: makeTraces(3, 'p2'), next_page_token: undefined },
    };
    renderPage();
    expect(await findTraceRow('p1-000')).toBeInTheDocument();

    await user.click(screen.getByLabelText('Next page'));
    expect(await findTraceRow('p2-000')).toBeInTheDocument();
    // The second search carried the recorded next-page token.
    expect(state.searchCalls.some((c) => c.page_token === 'token-2')).toBe(true);

    const callsBeforeBack = state.searchCalls.length;
    await user.click(screen.getByLabelText('Previous page'));
    expect(await findTraceRow('p1-000')).toBeInTheDocument();
    // Going back to a cached page issues no new request.
    expect(state.searchCalls.length).toBe(callsBeforeBack);
  });

  test('changing page size sends max_results and resets to page 1', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    state.pages = { '': { traces: makeTraces(3), next_page_token: 'token-2' } };
    renderPage();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();
    expect(state.searchCalls[0].max_results).toBe(25);

    await user.click(screen.getByLabelText('Rows per page'));
    await user.click(await screen.findByRole('option', { name: '100' }));

    await waitFor(() => expect(state.searchCalls.some((c) => c.max_results === 100)).toBe(true));
    expect(new URLSearchParams(env.lastSearch).get('pageSize')).toBe('100');
    expect(new URLSearchParams(env.lastSearch).get('page')).toBeNull();
  });

  describe('pagination bar', () => {
    test('is shown with Prev/Next disabled when there is only a single page of results', async () => {
      // No next_page_token means a single page: the bar still renders (page-size selector lives in
      // it) and both cursor buttons are disabled (Next via the last-page marker, Prev on page 1).
      state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');

      // Page-size selector is present…
      expect(screen.getByLabelText('Rows per page')).toBeInTheDocument();
      // …and Next is disabled on a single page (no next token), as is Prev on page 1.
      expect(screen.getByLabelText('Next page')).toBeDisabled();
      expect(screen.getByLabelText('Previous page')).toBeDisabled();
    });

    test('a partial final page disables Next even when the server sends a next_page_token', async () => {
      // Reproduces the reported bug: the long-running backend returns a real token on every non-empty
      // page including a partial final one. 21 rows at the default page size 25 is a short page, so the
      // frontend must treat it as the last page and disable Next — despite the non-empty token.
      state.pages = { '': { traces: makeTraces(21), next_page_token: 'phantom-token' } };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByLabelText('Next page')).toBeDisabled();
      expect(screen.getByLabelText('Previous page')).toBeDisabled();
    });

    test('an exactly-full last page → Next enabled; clicking it shows "No more results" with the bar still present', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      // Page 1 is exactly full (25 rows at pageSize 25) with a real token, so the client can't yet know
      // page 2 is empty — Next must stay enabled. Clicking Next fetches an empty page 2.
      state.pages = {
        '': { traces: makeTraces(25, 'p1'), next_page_token: 'token-2' },
        'token-2': { traces: [], next_page_token: undefined },
      };
      renderPage();
      expect(await findTraceRow('p1-000')).toBeInTheDocument();
      const next = screen.getByLabelText('Next page');
      expect(next).toBeEnabled();

      await user.click(next);

      // The distinct end-of-results state renders — NOT the initial "No traces yet" empty state…
      expect(await screen.findByText('No more results')).toBeInTheDocument();
      expect(screen.queryByText('No traces yet')).not.toBeInTheDocument();
      // …the pagination bar is still present (page-size selector lives in it)…
      expect(screen.getByLabelText('Rows per page')).toBeInTheDocument();
      // …Prev is enabled (back to page 1), Next is disabled (the empty page is terminal).
      expect(screen.getByLabelText('Previous page')).toBeEnabled();
      expect(screen.getByLabelText('Next page')).toBeDisabled();

      // Stepping back returns to page 1's rows (served from cache, no refetch needed).
      const callsBeforeBack = state.searchCalls.length;
      await user.click(screen.getByLabelText('Previous page'));
      expect(await findTraceRow('p1-000')).toBeInTheDocument();
      expect(state.searchCalls.length).toBe(callsBeforeBack);
    });

    test('shows the "{n} of {total}" count — current page rows out of the metrics total', async () => {
      // 3 rows on the page; the trace-metrics endpoint reports 42 total.
      state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
      env.metricsTotalCount = 42;
      renderPage();
      await findTraceRow('tr-000');

      expect(await screen.findByText('3 of 42')).toBeInTheDocument();
    });
  });
});
