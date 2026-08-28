import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { rest } from 'msw';
import { makeTrace, makeTraces } from '../test-utils/mockTraces';
import { TRACE_DENSITY_STORAGE_KEY_PREFIX } from '../utils/constants';
import { DENSITY_STORAGE_VERSION } from '../hooks/useTracesV4Density';
import { setLocalStorageItem } from '@databricks/web-shared/hooks';
import { slowlyTypeEachKey } from '@databricks/web-shared/test-utils/slowlyTypeEachKey';
import {
  renderPage,
  findTraceRow,
  queryTraceRow,
  pressEnter,
  openDisplaySubmenu,
  selectSubmenuItem,
  sortByTime,
  server,
  state,
  env,
  EXPERIMENT_ID,
  URL,
  SEARCH_ENDPOINT,
  type SearchCall,
} from '../test-utils/tracesV4PageContentTestBed';

// The saved-views hook (mounted via the toolbar's Views button) reads experiment tags through the
// Apollo experiment query. These page tests exercise the table/filter/pagination flows, not saved
// views, so stub it with a stable empty-tags result — leaving it unmocked fires a real Apollo query
// with no GraphQL handler, whose async failure slows the heavy filter-interaction tests past their
// timeout. The saved-views behavior itself is covered in TracesV4SavedViews.test.tsx.
jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => ({ data: { tags: [] }, refetch: () => Promise.resolve({}) }),
}));

describe('TracesV4PageContent (interactions)', () => {
  describe('filter', () => {
    const applyErrorStateClause = async (user: ReturnType<typeof userEvent.setup>) => {
      // Open the popover; the first row defaults to Field=State, Operator=`=`. Pick a state value.
      await user.click(screen.getByRole('button', { name: /Filters/ }));
      await user.click(await screen.findByLabelText('Filter value'));
      await user.click(await screen.findByRole('option', { name: 'Error' }));
      await user.click(screen.getByRole('button', { name: 'Apply filters' }));
      await waitFor(() =>
        expect(state.searchCalls.some((c) => c.filter?.includes("attributes.status = 'ERROR'"))).toBe(true),
      );
    };

    // Serve a filter-specific row set so a clear is observable in the DOM: an ERROR-filtered search
    // returns `err-000`, an unfiltered search returns `all-000`. (The default handler serves the same
    // rows regardless of filter, and React Query serves the unfiltered page from cache on clear — so
    // asserting on network calls can't distinguish "filter cleared" from "filter still applied".)
    const useFilterDistinctRows = () => {
      server.use(
        rest.post(SEARCH_ENDPOINT, async (req, res, ctx) => {
          const body = (await req.json()) as SearchCall;
          state.searchCalls.push(body);
          const hasError = body.filter?.includes("attributes.status = 'ERROR'");
          const id = hasError ? 'err-000' : 'all-000';
          return res(
            ctx.json({
              traces: [makeTrace(id, { state: hasError ? 'ERROR' : 'OK' })],
              next_page_token: undefined,
            }),
          );
        }),
      );
    };

    test('building a State clause and applying it sends the compiled status filter', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');

      await applyErrorStateClause(user);
      expect(state.searchCalls.some((c) => c.filter?.includes("attributes.status = 'ERROR'"))).toBe(true);
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    // TODO(traces-v4): service_name filtering was dropped in OSS (the SearchTracesV3 parser rejects span.service_name). Field + clause removed by design.

    test.skip('building a service name clause and applying it sends the compiled span.service_name filter', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');

      // Open the popover, change filter field to service name, type a value, and apply.
      await user.click(screen.getByRole('button', { name: /Filters/ }));
      await user.click(await screen.findByLabelText('Filter field'));
      await user.click(await screen.findByRole('option', { name: 'Service name' }));
      await slowlyTypeEachKey(await screen.findByLabelText('Filter value'), 'my-service', { user });
      await user.click(screen.getByRole('button', { name: 'Apply filters' }));

      await waitFor(() =>
        expect(state.searchCalls.some((c) => c.filter?.includes("span.service_name = 'my-service'"))).toBe(true),
      );
    });

    test('the clear-all button clears an applied clause and the unfiltered rows return', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      useFilterDistinctRows();
      renderPage();
      expect(await findTraceRow('all-000')).toBeInTheDocument();

      await applyErrorStateClause(user);
      // The filtered result set is shown and the count badge lights up.
      expect(await findTraceRow('err-000')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Filters/ })).toHaveTextContent('(1)');

      // Clear-all is a real, keyboard-reachable button (a sibling of the trigger, not a nested icon),
      // so it's discoverable by role/name — clicking it drops the clause, its badge, and the filter.
      await user.click(screen.getByRole('button', { name: 'Clear all filters' }));

      expect(await findTraceRow('all-000')).toBeInTheDocument();
      expect(queryTraceRow('err-000')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Filters/ })).not.toHaveTextContent('(1)');
      // Multi-step filter interaction on a full page render — bump off the flaky 5s default.
    }, 20000);

    test('the clear-all button also clears an applied click-to-filter tag', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTrace('tr-000', { tags: { env: 'prod', team: 'ml' } })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      // A tag filter (a URL-backed concept, distinct from the popover clauses) counts toward the badge.
      await user.click(screen.getByRole('button', { name: 'Filter by tag env: prod' }));
      await waitFor(() => expect(new URLSearchParams(env.lastSearch).getAll('tag')).toContain('env=prod'));
      expect(screen.getByRole('button', { name: /Filters/ })).toHaveTextContent('(1)');

      // Clear-all must clear the tag param too (not only the popover clauses), or the badge would stay
      // lit and the filter would remain applied — the reported bug.
      await user.click(screen.getByRole('button', { name: 'Clear all filters' }));

      expect(new URLSearchParams(env.lastSearch).getAll('tag')).not.toContain('env=prod');
      expect(screen.getByRole('button', { name: /Filters/ })).not.toHaveTextContent('(1)');
      // Multi-step tag-filter interaction on a full page render — bump off the flaky 5s default.
    }, 20000);

    test('the popover Clear filters button clears the applied clause and the unfiltered rows return', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      useFilterDistinctRows();
      renderPage();
      expect(await findTraceRow('all-000')).toBeInTheDocument();

      await applyErrorStateClause(user);
      expect(await findTraceRow('err-000')).toBeInTheDocument();

      // Reopen the popover and use its Clear button (distinct from removing a single clause row).
      await user.click(screen.getByRole('button', { name: /Filters/ }));
      await user.click(await screen.findByRole('button', { name: 'Clear filters' }));

      expect(await findTraceRow('all-000')).toBeInTheDocument();
      expect(queryTraceRow('err-000')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Filters/ })).not.toHaveTextContent('(1)');
      // Filter apply + clear each re-render the heavy table; per-test timeout avoids the
      // lint-forbidden global jest.setTimeout under parallel jsdom load.
    }, 20000);
  });

  describe('default time range', () => {
    test('displays the 7 day range on mount when the URL has no startTimeLabel', async () => {
      renderPage();
      await findTraceRow('tr-000');
      // The standalone hook resolves the default without writing it to the URL, so assert the
      // dropdown *displays* the default rather than checking the URL param.
      expect(screen.getByRole('button', { name: 'Time range: Last 7 days' })).toHaveTextContent(/^Last 7 days$/);
    });

    test('ignores a v3 saved time selection (isolated v4 localStorage key)', async () => {
      // Seed the legacy v3 key exactly as the shared useMonitoringFilters persists it (version 1,
      // scoped). v4 uses a distinct key, so it must not read this — the dropdown should still show the
      // v4 default.
      setLocalStorageItem(`traces_useMonitoringFilters_${EXPERIMENT_ID}`, 1, true, { startTimeLabel: 'LAST_30_DAYS' });
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByRole('button', { name: 'Time range: Last 7 days' })).toHaveTextContent(/^Last 7 days$/);
    });
  });

  describe('URL preservation (v3-compatible params)', () => {
    test('a full deep-link drives the correct initial request and the params survive on the URL', async () => {
      state.pages = {
        // page=2 needs a reachable cursor; but the token cache is memory-only, so on a fresh load the
        // stale-page recovery resets to page 1 (documented behavior). The request assertions below
        // target the filter/order_by/max_results the URL drives, which are page-independent.
        '': { traces: makeTraces(3), next_page_token: undefined },
      };
      renderPage({
        initialUrl: `${URL}?q=hello&pageSize=100&sort=duration&dir=asc&startTimeLabel=LAST_7_DAYS&tag=env%3Dprod`,
      });
      await findTraceRow('tr-000');

      const firstCall = state.searchCalls[0];
      // Search → ILIKE; tag → compiled tags clause; time-range label → ms bounds on timestamp_ms.
      expect(firstCall.filter).toContain("trace.text ILIKE '%hello%'");
      expect(firstCall.filter).toContain("tags.env = 'prod'");
      expect(firstCall.filter).toContain('attributes.timestamp_ms >');
      expect(firstCall.filter).toContain('attributes.timestamp_ms <');
      // Sort → order_by; pageSize → max_results.
      expect(firstCall.order_by?.[0]).toBe('execution_time ASC');
      expect(firstCall.max_results).toBe(100);

      // The shared/v4 params survive on the URL after mount.
      const search = new URLSearchParams(env.lastSearch);
      expect(search.get('q')).toBe('hello');
      expect(search.get('pageSize')).toBe('100');
      expect(search.get('sort')).toBe('duration');
      expect(search.get('dir')).toBe('asc');
      expect(search.get('startTimeLabel')).toBe('LAST_7_DAYS');
      expect(search.getAll('tag')).toContain('env=prod');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('the shared v3 CUSTOM time bounds are honored on mount and preserved across an in-tab action', async () => {
      const start = '2025-01-01T00:00:00.000Z';
      const end = '2025-01-02T00:00:00.000Z';
      const user = userEvent.setup();
      renderPage({
        initialUrl: `${URL}?startTimeLabel=CUSTOM&startTime=${encodeURIComponent(start)}&endTime=${encodeURIComponent(
          end,
        )}`,
      });
      await findTraceRow('tr-000');

      // The explicit CUSTOM bounds map to the ms timestamps on the initial request.
      const startMs = new Date(start).getTime();
      const endMs = new Date(end).getTime();
      expect(state.searchCalls[0].filter).toContain(`attributes.timestamp_ms > ${startMs}`);
      expect(state.searchCalls[0].filter).toContain(`attributes.timestamp_ms < ${endMs}`);

      // An in-tab action (search) issues a new request that still carries the same time bounds…
      // The search commits on Enter, so type then press Enter.
      const searchBox = screen.getByPlaceholderText('Search traces by id, input, or output');
      await slowlyTypeEachKey(searchBox, 'x', { user });
      pressEnter(searchBox);
      await waitFor(() =>
        expect(
          state.searchCalls.some(
            (c) => c.filter?.includes(`attributes.timestamp_ms > ${startMs}`) && c.filter?.includes("ILIKE '%x%'"),
          ),
        ).toBe(true),
      );
      // …and the shared v3 time params are preserved on the URL.
      const search = new URLSearchParams(env.lastSearch);
      expect(search.get('startTimeLabel')).toBe('CUSTOM');
      expect(search.get('startTime')).toBe(start);
      expect(search.get('endTime')).toBe(end);
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('a time-range change issues a new request with new time bounds', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');
      const callsBefore = state.searchCalls.length;

      // Open the time-range dropdown and pick a different preset.
      await user.click(screen.getByRole('button', { name: 'Time range: Last 7 days' }));
      await user.click(await screen.findByRole('option', { name: 'Last 15 minutes' }));

      // A new search fires (the filter's time bounds changed), so the call count grows.
      await waitFor(() => expect(state.searchCalls.length).toBeGreaterThan(callsBefore));
      expect(new URLSearchParams(env.lastSearch).get('startTimeLabel')).toBe('LAST_15_MINUTES');
      expect(screen.getByRole('button', { name: 'Time range: Last 15 minutes' })).toHaveTextContent(
        /^Last 15 minutes$/,
      );
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('selecting a custom value shows the absolute range picker and preset button', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await findTraceRow('tr-000');

      await user.click(screen.getByRole('button', { name: 'Time range: Last 7 days' }));
      await user.click(await screen.findByRole('option', { name: 'Custom' }));

      await waitFor(() => expect(new URLSearchParams(env.lastSearch).get('startTimeLabel')).toBe('CUSTOM'));
      expect(screen.getAllByRole('textbox', { name: 'Select Date and Time' })).toHaveLength(2);
      expect(screen.getByRole('button', { name: 'Time range' })).toBeInTheDocument();

      // The calendar button keeps the preset choices (without a redundant Custom option) and
      // returns to the compact preset control after a selection.
      await user.click(screen.getByRole('button', { name: 'Time range' }));
      expect(screen.queryByRole('option', { name: 'Custom' })).not.toBeInTheDocument();
      await user.click(await screen.findByRole('option', { name: 'Last 7 days' }));
      expect(screen.getByRole('button', { name: 'Time range: Last 7 days' })).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('trace actions ("Use for evaluation" group)', () => {
    // Opens the Actions dropdown after selecting the first trace, returning the menu for scoping.
    const openActionsMenuForFirstTrace = async (user: ReturnType<typeof userEvent.setup>) => {
      await findTraceRow('tr-000');
      await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-000' }));
      await user.click(await screen.findByRole('button', { name: 'Actions for selected traces' }));
      return screen.findByRole('menu');
    };

    test('always offers "Run scorers" and "Add to evaluation dataset" for a selection', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      const menu = await openActionsMenuForFirstTrace(user);

      // These two are always available (no extra feature gate) — the core evaluation actions.
      expect(within(menu).getByRole('menuitem', { name: 'Run scorers' })).toBeInTheDocument();
      expect(within(menu).getByRole('menuitem', { name: 'Add to evaluation dataset' })).toBeInTheDocument();
      // Delete remains, below the evaluation group. In OSS traces have no UC gate, so it's enabled.
      const deleteItem = within(menu).getByRole('menuitem', { name: /Delete/ });
      expect(deleteItem).toBeInTheDocument();
      expect(deleteItem).not.toHaveAttribute('aria-disabled', 'true');
    }, 20000);

    test('offers "Flag for review" — review queues ship in OSS, matching v3', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      const menu = await openActionsMenuForFirstTrace(user);

      // v3 (TracesV3Logs) wires AddToReviewQueueDropdown with addToReviewQueue: true in OSS, and OSS
      // ships a full review-queue page/route, so "Flag for review" must be offered in v4 too.
      expect(within(menu).getByRole('menuitem', { name: 'Flag for review' })).toBeInTheDocument();
    }, 20000);

    test('offers "Compare" and "Edit tags", gated on the selection size (v3 parity)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      const menu = await openActionsMenuForFirstTrace(user);

      // With one trace selected: Edit tags is actionable, Compare needs 2–3 (both present, matching v3).
      const editTags = within(menu).getByRole('menuitem', { name: 'Edit tags' });
      expect(editTags).not.toHaveAttribute('aria-disabled', 'true');
      const compare = within(menu).getByRole('menuitem', { name: 'Compare' });
      expect(compare).toHaveAttribute('aria-disabled', 'true');
    }, 20000);

    test('opening "Run scorers" launches the scorer-selection modal for the selected trace', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      const menu = await openActionsMenuForFirstTrace(user);
      await user.click(within(menu).getByRole('menuitem', { name: 'Run scorers' }));

      // The shared run-judges modal opens scoped to the single selected trace. (The menu item reads
      // "Run scorers"; the modal title uses the underlying "judge" terminology.)
      expect(await screen.findByRole('dialog', { name: /Run judge on trace/ })).toBeInTheDocument();
    }, 30000); // heavy full-page userEvent render + modal — bump off the flaky default under parallel jsdom load

    test('bulk actions run on the full cross-page selection (select on page 1, page to 2, Run scorers)', async () => {
      // The selection now stores each trace's full info keyed by id, so it spans pages. Selecting one
      // trace on page 1, paging to page 2, selecting another there, then running scorers must scope the
      // action to BOTH selected traces — not just the page-2 subset (the reported "Run scorers (2)" runs
      // on 0/1 traces bug).
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      // Exactly pageSize (25) rows so Next enables (rowCount < pageSize marks the last page); a bigger
      // page just slows the full render without changing the cross-page selection being asserted.
      state.pages = {
        '': { traces: makeTraces(25, 'p1'), next_page_token: 'token-2' },
        'token-2': { traces: makeTraces(25, 'p2'), next_page_token: undefined },
      };
      renderPage();
      expect(await findTraceRow('p1-000')).toBeInTheDocument();

      // Select a trace on page 1, then paginate to page 2 (the selection persists across the page swap).
      await user.click(screen.getByRole('checkbox', { name: 'Select trace p1-000' }));
      await user.click(screen.getByRole('button', { name: 'Next page' }));
      expect(await findTraceRow('p2-000')).toBeInTheDocument();

      // Select a trace on page 2 — now the cross-page selection is {p1-000, p2-000}.
      await user.click(screen.getByRole('checkbox', { name: 'Select trace p2-000' }));

      const actionsButton = await screen.findByRole('button', { name: 'Actions for selected traces' });
      expect(actionsButton).toHaveTextContent('Actions (2)');
      await user.click(actionsButton);
      await user.click(within(await screen.findByRole('menu')).getByRole('menuitem', { name: 'Run scorers' }));

      // The scorer modal launches for the full 2-trace cross-page selection, not the page-2 subset.
      expect(await screen.findByRole('dialog', { name: /Run judge on 2 traces/ })).toBeInTheDocument();
    }, 60000); // heavy: renders two 50-row pages, selects across a page swap, then opens the modal
  });

  describe('trace drawer navigation', () => {
    // The drawer binds ArrowLeft/ArrowRight at the window level; driving it via the keyboard is the
    // user-faithful way to exercise nav (the header's chevron buttons are icon-only, no a11y name).
    test('ArrowRight advances to the next row, writing its V4 long id to the URL', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));
      await screen.findByRole('dialog');
      expect(new URLSearchParams(env.lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-000');

      await user.keyboard('{ArrowRight}');
      // Advancing stays a V4 long id (not a bare hex id, which would hit the legacy path).
      await waitFor(() => expect(new URLSearchParams(env.lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-001'));
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('ArrowLeft on the first row is a no-op (Back disabled at the page start)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));
      await screen.findByRole('dialog');

      await user.keyboard('{ArrowLeft}');
      // Still on the first row — there's no previous trace on the page.
      expect(new URLSearchParams(env.lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-000');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('ArrowRight on the last row of the page is a no-op (Next disabled at the page end)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      // The default 3-trace fixture ends at tr-002.
      await user.click(await findTraceRow('tr-002'));
      await screen.findByRole('dialog');

      await user.keyboard('{ArrowRight}');
      expect(new URLSearchParams(env.lastSearch).get('traceId')).toBe('trace:/cat.sch/tr-002');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('toolbar order', () => {
    // Assert the always-present V4 control ordering. Views pins far left (before the date selector);
    // column/sort/row-height controls consolidate into the Display popover; Refresh then Detect Issues
    // pin far right (Detect Issues renders whenever issue detection is enabled, true in OSS).
    test('renders Views → Date → Search → Filter → Display → Refresh → Detect Issues', async () => {
      renderPage();
      await findTraceRow('tr-000');

      const views = screen.getByTestId('trace-v4-saved-views-trigger');
      const date = screen.getByTestId('time-range-select-dropdown');
      const search = screen.getByPlaceholderText('Search traces by id, input, or output');
      const filter = screen.getByRole('button', { name: /Filters/ });
      const display = screen.getByRole('button', { name: 'Display' });
      const detectIssues = screen.getByRole('button', { name: 'Detect issues in traces' });
      // The refresh button's accessible name is a relative-time label ("now" / "1 second ago") that
      // drifts with render time, so match it by its stable componentId instead.
      const refresh = document.querySelector('[data-component-id="mlflow.traces-v4.refresh-date-button"]')!;
      // DOCUMENT_POSITION_FOLLOWING (4) means the arg node comes after `this` node in document order.
      expect(views.compareDocumentPosition(date) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(date.compareDocumentPosition(search) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(search.compareDocumentPosition(filter) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(filter.compareDocumentPosition(display) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(display.compareDocumentPosition(refresh) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(refresh.compareDocumentPosition(detectIssues) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      // The full page render (incl. the Detect Issues button's first-visit guidance popover) is slow
      // under parallel jsdom load; per-test timeout avoids the lint-forbidden global jest.setTimeout.
    }, 20000);

    // TODO(traces-v4): a "clicking Detect Issues opens the modal" test would drive the shared
    // IssueDetectionModal, which fires gateway endpoint + API-key queries this suite's MSW doesn't
    // stub (it hangs). The button→modal wiring mirrors v3's; presence/order is covered above.

    // TODO(traces-v4): The Actions button only renders on selection, which is jsdom-blocked here (portalled DuBois checkbox).
    test.skip('places the selection Actions button between Display and Refresh', async () => {
      const user = userEvent.setup();
      renderPage();
      await findTraceRow('tr-000');
      await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-000' }));

      const display = screen.getByRole('button', { name: 'Display' });
      const actions = await screen.findByRole('button', { name: 'Actions for selected traces' });
      const refresh = screen.getByRole('button', { name: 'now' });
      expect(display.compareDocumentPosition(actions) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(actions.compareDocumentPosition(refresh) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }, 20000);
  });

  describe('Display popover: sort', () => {
    test('choosing Duration (descending) sends a duration order_by', async () => {
      const user = userEvent.setup();
      renderPage();
      await findTraceRow('tr-000');

      await openDisplaySubmenu(user, /^Sort/);
      await selectSubmenuItem('menuitemradio', 'Duration (descending)');

      // buildOrderBy maps the duration column to `execution_time DESC`; the new search carries it.
      await waitFor(() =>
        expect(state.searchCalls.some((c) => c.order_by?.[0]?.startsWith('execution_time'))).toBe(true),
      );
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('reflects the active sort (Time, descending) as the checked radio', async () => {
      const user = userEvent.setup();
      renderPage();
      await findTraceRow('tr-000');

      await openDisplaySubmenu(user, /^Sort/);
      // The default sort is start_time/desc, so the "Time (descending)" radio is the checked one.
      expect(
        await screen.findByRole('menuitemradio', { name: 'Time (descending)', checked: true }),
      ).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('Display popover: row height', () => {
    // Standard and Tall keep the same row padding; their row height differs by the density minHeight
    // floor and preview line clamp.
    const STANDARD_ROW_PADDING = '6px';
    const STANDARD_ROW_MIN_HEIGHT = '48px';
    const TALL_ROW_MIN_HEIGHT = '128px';
    const firstTraceRow = () => screen.getByRole('row', { name: /Open trace tr-000 — input/ });

    test('defaults to Compact when the stored preference uses the previous version', async () => {
      const user = userEvent.setup();
      const densityKey = `${TRACE_DENSITY_STORAGE_KEY_PREFIX}.${EXPERIMENT_ID}`;
      // A stale (previous-version) entry is ignored, so density falls back to the Compact default.
      setLocalStorageItem(densityKey, DENSITY_STORAGE_VERSION - 1, true, 'tall');

      renderPage();
      await findTraceRow('tr-000');

      await openDisplaySubmenu(user, /^Row height/);
      expect(await screen.findByRole('menuitemradio', { name: 'Compact', checked: true })).toBeInTheDocument();
    });

    test('switching to Standard makes the table use standard-height rows with a larger minimum height', async () => {
      const user = userEvent.setup();
      renderPage();
      await findTraceRow('tr-000');

      await openDisplaySubmenu(user, /^Row height/);
      await selectSubmenuItem('menuitemradio', 'Standard');

      await waitFor(() =>
        expect(firstTraceRow()).toHaveStyle({
          '--table-row-vertical-padding': STANDARD_ROW_PADDING,
          minHeight: STANDARD_ROW_MIN_HEIGHT,
        }),
      );
    });

    test('switching to Tall keeps standard table padding and uses a taller row height', async () => {
      const user = userEvent.setup();
      renderPage();
      await findTraceRow('tr-000');

      await openDisplaySubmenu(user, /^Row height/);
      await selectSubmenuItem('menuitemradio', 'Tall');

      await waitFor(() =>
        expect(firstTraceRow()).toHaveStyle({
          '--table-row-vertical-padding': STANDARD_ROW_PADDING,
          minHeight: TALL_ROW_MIN_HEIGHT,
        }),
      );
    });

    test('applies a Tall row height persisted from a prior session', async () => {
      const user = userEvent.setup();
      // A density chosen in a prior session (Tall) is read back on the next mount via
      // the same scoped/versioned key the hook writes — the table uses standard rows and the Row height
      // submenu shows Tall as the checked option.
      const densityKey = `${TRACE_DENSITY_STORAGE_KEY_PREFIX}.${EXPERIMENT_ID}`;
      setLocalStorageItem(densityKey, DENSITY_STORAGE_VERSION, true, 'tall');

      renderPage();
      await findTraceRow('tr-000');
      expect(firstTraceRow()).toHaveStyle({
        '--table-row-vertical-padding': STANDARD_ROW_PADDING,
        minHeight: TALL_ROW_MIN_HEIGHT,
      });

      await openDisplaySubmenu(user, /^Row height/);
      expect(await screen.findByRole('menuitemradio', { name: 'Tall', checked: true })).toBeInTheDocument();
    });
  });
});
