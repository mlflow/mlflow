import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { rest } from 'msw';
import {
  makeFeedbackAssessment,
  makeSessionTrace,
  makeTaggedTrace,
  makeTrace,
  makeTraces,
} from '../test-utils/mockTraces';
import { TRACE_COLUMN_SIZES_STORAGE_KEY_PREFIX, TRACE_DENSITY_STORAGE_KEY_PREFIX } from '../utils/constants';
import { COLUMN_SIZES_STORAGE_VERSION } from '../hooks/useTracesV4ColumnSizing';
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

describe('TracesV4PageContent', () => {
  test('renders a row per trace with the default columns', async () => {
    renderPage();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();
    expect(screen.getByText('request for tr-000')).toBeInTheDocument();
    expect(screen.getByText('response for tr-000')).toBeInTheDocument();
    // Default columns: Time, Input, Output, Duration, State, Tokens (no Session on this session-less page).
    expect(screen.getByRole('columnheader', { name: 'Time' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'State' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Tokens' })).toBeInTheDocument();
    // Opt-in columns are hidden by default (available via the column selector).
    expect(screen.queryByRole('columnheader', { name: 'Trace ID' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Trace name' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'User' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Source' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Run name' })).not.toBeInTheDocument();
    // Per-test timeout (matching the file's other heavy renders): the full page render is slow under
    // parallel jsdom load and would otherwise flake against the default 5s ceiling.
  }, 20000);

  test('enabling Trace ID puts it as the first data column, left of Time', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    await openDisplaySubmenu(user, /^Columns/);
    await selectSubmenuItem('menuitemcheckbox', 'Trace ID');

    const traceId = await screen.findByRole('columnheader', { name: 'Trace ID' });
    const time = screen.getByRole('columnheader', { name: 'Time' });
    // DOCUMENT_POSITION_FOLLOWING (4) means Time comes after Trace ID in document order — i.e. Trace
    // ID is the leftmost data column (left of Time). The row-select cell renders before both.
    expect(traceId.compareDocumentPosition(time) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  }, 20000);

  test('offers the restored legacy columns in the column selector', async () => {
    const user = userEvent.setup();
    renderPage();
    await findTraceRow('tr-000');

    await openDisplaySubmenu(user, /^Columns/);
    expect(await screen.findByRole('menuitemcheckbox', { name: 'Trace name' })).toBeInTheDocument();
    expect(screen.getByRole('menuitemcheckbox', { name: 'User' })).toBeInTheDocument();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Source' })).toBeInTheDocument();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Run name' })).toBeInTheDocument();
  });

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

    await user.click(screen.getByRole('button', { name: 'Next page' }));
    expect(await findTraceRow('p2-000')).toBeInTheDocument();
    // The second search carried the recorded next-page token.
    expect(state.searchCalls.some((c) => c.page_token === 'token-2')).toBe(true);

    const callsBeforeBack = state.searchCalls.length;
    await user.click(screen.getByRole('button', { name: 'Previous page' }));
    expect(await findTraceRow('p1-000')).toBeInTheDocument();
    // Going back to a cached page issues no new request.
    expect(state.searchCalls.length).toBe(callsBeforeBack);
    // Two full renders of the heavy shared table (Next then Prev) push this past the 5s default in
    // jsdom; a per-test timeout keeps it green without the lint-forbidden global jest.setTimeout.
  }, 20000);

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
    // Re-render on the page-size change is slow in jsdom; per-test timeout avoids global jest.setTimeout.
  }, 20000);

  test('sorting by Time sends an order_by and resets the page', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();

    await sortByTime(user);
    // The initial load sorts timestamp DESC; the ascending pick must issue a distinct 'timestamp ASC'.
    await waitFor(() => expect(state.searchCalls.some((c) => c.order_by?.[0] === 'timestamp ASC')).toBe(true));
  });

  test('clicking the active sort icon in a column header flips the sort direction', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    expect(await findTraceRow('tr-000')).toBeInTheDocument();

    // Time is the default sort (descending), so its header shows a sort-direction icon. That icon is
    // itself clickable now (previously only the "Column options" chevron sorted) and flips to asc.
    const timeHeader = screen.getByRole('columnheader', { name: 'Time' });
    await user.click(within(timeHeader).getByRole('button', { name: /^Sort/ }));
    await waitFor(() => expect(state.searchCalls.some((c) => c.order_by?.[0] === 'timestamp ASC')).toBe(true));
  });

  test('non-sortable headers (State) expose no sort control in their menu', async () => {
    const user = userEvent.setup();
    renderPage();
    await findTraceRow('tr-000');

    const stateHeader = screen.getByRole('columnheader', { name: 'State' });
    // A display-only column still has a menu (for Hide column) but offers no sort items.
    await user.click(within(stateHeader).getByRole('button', { name: 'Column options' }));
    expect(await screen.findByRole('menuitem', { name: 'Hide column' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: /^Sort/ })).not.toBeInTheDocument();
  });

  test('select rows → Actions (2) → Delete is enabled in OSS and opens the confirm modal', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-000' }));
    await user.click(screen.getByRole('checkbox', { name: 'Select trace tr-001' }));

    // The Actions button's accessible name is its aria-label; the "(2)" count is visible text.
    const actionsButton = await screen.findByRole('button', { name: 'Actions for selected traces' });
    expect(actionsButton).toHaveTextContent('Actions (2)');
    await user.click(actionsButton);

    // OSS traces have no UC gate (isDeleteDisabled === false), so Delete is enabled and clicking it
    // opens the delete-confirmation modal (mirrors the datasets-v2 delete flow).
    const deleteItem = await screen.findByRole('menuitem', { name: /Delete/ });
    expect(deleteItem).not.toHaveAttribute('aria-disabled', 'true');
    await user.click(deleteItem);
    expect(await screen.findByRole('dialog')).toHaveTextContent(/Delete/);
  }, 20000);

  test('searching commits only on Enter, not per keystroke, then sends an ILIKE filter', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    // Type key-by-key (the sanctioned escape hatch for the no-userevent-type lint rule). Typing alone
    // must NOT fire the search — the box commits on Enter now, not per keystroke.
    const searchBox = screen.getByPlaceholderText('Search traces by id, input, or output');
    await slowlyTypeEachKey(searchBox, 'hello', { user });
    expect(state.searchCalls.some((c) => c.filter?.includes("trace.text ILIKE '%hello%'"))).toBe(false);

    // Pressing Enter commits the typed value and the ILIKE search fires.
    pressEnter(searchBox);
    await waitFor(() =>
      expect(state.searchCalls.some((c) => c.filter?.includes("trace.text ILIKE '%hello%'"))).toBe(true),
    );
  });

  test('clearing the search (X) commits an empty search immediately', async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPage();
    await findTraceRow('tr-000');

    // Commit a search first (Enter), then clear it.
    const searchBox = screen.getByPlaceholderText('Search traces by id, input, or output');
    await slowlyTypeEachKey(searchBox, 'hello', { user });
    pressEnter(searchBox);
    await waitFor(() => expect(new URLSearchParams(env.lastSearch).get('q')).toBe('hello'));

    // Clicking the input's clear (X) affordance commits the empty search right away (no Enter needed).
    // Du Bois Input renders the X with the accessible name `close-circle` (matches the datasets-v2 tests).
    await user.click(screen.getByLabelText('close-circle'));
    await waitFor(() => expect(new URLSearchParams(env.lastSearch).get('q')).toBeNull());
  });

  // TODO(traces-v4): The OSS empty state renders TracesViewTableNoTracesQuickstart; this test asserts the Databricks TracingQuickStart CTA copy. Rewrite for the OSS quickstart.

  test.skip('shows the tracing quickstart CTA when the experiment has no traces', async () => {
    state.pages = { '': { traces: [], next_page_token: undefined } };
    renderPage();
    // The unmocked experiment query surfaces no kind, so the non-GenAI generic quickstart renders
    // ("No traces recorded") rather than the old generic shared "No traces yet" empty state.
    expect(await screen.findByText('No traces recorded')).toBeInTheDocument();
    expect(screen.queryByText('No traces yet')).not.toBeInTheDocument();
  });

  describe('robustness', () => {
    test('a search error shows an error state with a working Retry', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.searchShouldFail = true;
      renderPage();
      expect(await screen.findByText("Couldn't load traces")).toBeInTheDocument();

      // Recover: next fetch succeeds.
      state.searchShouldFail = false;
      await user.click(screen.getByRole('button', { name: 'Retry' }));
      expect(await findTraceRow('tr-000')).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('a first-load error surfaces the server message (not the generic fallback)', async () => {
      // A permission error must read distinctly from a timeout: the raw server message is shown so the
      // cause isn't discarded (the reported regression vs v3, where every error looked identical).
      state.searchShouldFail = true;
      state.searchErrorMessage = 'PERMISSION_DENIED: user lacks access to the traces table';
      renderPage();
      expect(await screen.findByText("Couldn't load traces")).toBeInTheDocument();
      expect(screen.getByText(/PERMISSION_DENIED: user lacks access to the traces table/)).toBeInTheDocument();
    });

    test('a first-load SQL-warehouse timeout shows the larger-warehouse hint', async () => {
      // A SQL-warehouse timeout is detected (via the shared isSqlWarehouseTimeoutError) and gets the
      // actionable "try a larger warehouse" hint, matching v3 — not the raw message or generic text.
      state.searchShouldFail = true;
      state.searchErrorMessage = 'Timeout while issuing SQL query to fetch traces';
      renderPage();
      expect(await screen.findByText("Couldn't load traces")).toBeInTheDocument();
      expect(screen.getByText(/try selecting a larger SQL warehouse/)).toBeInTheDocument();
    });

    test('a malformed trace row (missing assessments, unparseable metadata) renders without throwing', async () => {
      state.pages = {
        '': {
          traces: [
            makeTrace('tr-bad', { assessments: undefined, trace_metadata: { 'mlflow.trace.tokenUsage': 'not json' } }),
          ],
          next_page_token: undefined,
        },
      };
      renderPage();
      // Row still renders; the bad token metadata simply shows nothing rather than crashing.
      expect(await findTraceRow('tr-bad')).toBeInTheDocument();
    });
  });

  describe('layout stability', () => {
    test('first load renders the real header plus exactly pageSize skeleton rows', async () => {
      // Delay the search so the skeleton is observable.
      let resolveSearch: (() => void) | undefined;
      server.use(
        rest.post(SEARCH_ENDPOINT, async (_req, res, ctx) => {
          await new Promise<void>((r) => {
            resolveSearch = r;
          });
          return res(ctx.json({ traces: makeTraces(3), next_page_token: undefined }));
        }),
      );
      renderPage();
      // Header is real from the first paint.
      expect(await screen.findByRole('columnheader', { name: 'Time' })).toBeInTheDocument();
      // Skeleton status region is present during first load.
      expect(screen.getByRole('region', { name: 'Traces' })).toHaveAttribute('aria-busy', 'true');
      resolveSearch?.();
      expect(await findTraceRow('tr-000')).toBeInTheDocument();
    });

    test('reloading shows the skeleton again (keyed off isFetching, not just first load)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      expect(await findTraceRow('tr-000')).toBeInTheDocument();

      // Make the *next* search hang so the reload's in-flight state is observable. A sort change
      // reuses keepPreviousData (isLoading stays false, isFetching flips true) — the exact reload
      // scenario that must now render the skeleton rather than keep the stale rows.
      let resolveReload: (() => void) | undefined;
      server.use(
        rest.post(SEARCH_ENDPOINT, async (req, res, ctx) => {
          const body = (await req.json()) as SearchCall;
          state.searchCalls.push(body);
          await new Promise<void>((r) => {
            resolveReload = r;
          });
          return res(ctx.json({ traces: makeTraces(3), next_page_token: undefined }));
        }),
      );

      await sortByTime(user);

      // Skeleton replaces rows (not kept via keepPreviousData).
      await waitFor(() => expect(screen.getByRole('region', { name: 'Traces' })).toHaveAttribute('aria-busy', 'true'));
      expect(queryTraceRow('tr-000')).not.toBeInTheDocument();

      resolveReload?.();
      expect(await findTraceRow('tr-000')).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('opening a trace', () => {
    test('input cells link to the V4 long identifier', async () => {
      renderPage();
      const link = await findTraceRow('tr-000');
      expect(link).toHaveAttribute('href', expect.stringContaining('traceId=trace%3A%2Fcat.sch%2Ftr-000'));
    });

    test('the trace link preserves the current URL params (filters survive opening a trace)', async () => {
      // Land on the page with an existing filter param already on the URL; opening a trace should
      // add `traceId` without dropping it, so a filtered view (or a Cmd/Ctrl+click into a new tab)
      // reopens with the same filters rather than resetting to a bare Traces route.
      const filterParam = new URLSearchParams({ filter: "trace.status = 'OK'" }).toString();
      renderPage({ initialUrl: `${URL}?${filterParam}` });
      const link = await findTraceRow('tr-000');
      const href = link.getAttribute('href') ?? '';
      expect(href).toContain('traceId=trace%3A%2Fcat.sch%2Ftr-000');
      expect(href).toContain(filterParam);
    });

    test('clicking a UC-backed row opens the drawer with a V4 long identifier (not a bare hex id)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));

      // The mock trace_location is UC_SCHEMA cat.sch, so the URL must carry the V4 long id
      // `trace:/cat.sch/tr-000` — the bare hex id would hit the legacy path and be rejected as
      // "Invalid request id". The URL long id also keeps the row-selection highlight matching.
      const longId = 'trace:/cat.sch/tr-000';
      const drawer = await screen.findByRole('dialog');
      expect(drawer).toBeInTheDocument();
      expect(new URLSearchParams(env.lastSearch).get('traceId')).toBe(longId);
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('the drawer header shows the trace id as a copyable tag, not the raw long id', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderPage();
      await user.click(await findTraceRow('tr-000'));

      // The redesigned (v2) header titles the drawer "Trace" and shows the id as a copyable tag (the
      // `tr-` prefix stripped, truncated to 8 chars) — not the long id `trace:/cat.sch/tr-000`.
      const drawer = await screen.findByRole('dialog');
      expect(await within(drawer).findByRole('button', { name: '000' })).toBeInTheDocument();
      expect(within(drawer).queryByText('trace:/cat.sch/tr-000')).not.toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load
  });

  describe('session column', () => {
    test('is visible by default when the page has session-tagged traces', async () => {
      state.pages = { '': { traces: [makeSessionTrace('tr-000')], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByRole('columnheader', { name: 'Session' })).toBeInTheDocument();
    });

    test('is hidden by default when no trace on the page has a session', async () => {
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.queryByRole('columnheader', { name: 'Session' })).not.toBeInTheDocument();
    });

    test('an explicit toggle sticks even on a page without sessions', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const { unmount } = renderPage();
      await findTraceRow('tr-000');

      // Turn Session on via Display → Columns, even though this page has no sessions.
      await openDisplaySubmenu(user, /^Columns/);
      await selectSubmenuItem('menuitemcheckbox', 'Session');
      expect(await screen.findByRole('columnheader', { name: 'Session' })).toBeInTheDocument();

      // Remount: the explicit "on" choice persists (a sticky override, not the data-driven default).
      unmount();
      renderPage();
      expect(await screen.findByRole('columnheader', { name: 'Session' })).toBeInTheDocument();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('the session value links to the chat-session page with the trace deep-linked', async () => {
      state.pages = { '': { traces: [makeSessionTrace('tr-000', 'sess-1')], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');

      // The session tag is wrapped in a link to the single-chat-session route (like v1), carrying the
      // current trace via ?selectedTraceId so the destination opens on that trace.
      const link = screen.getByRole('link', { name: 'sess-1' });
      const href = link.getAttribute('href') ?? '';
      expect(href).toContain('/experiments/exp-1/chat-sessions/sess-1');
      expect(href).toContain('selectedTraceId=tr-000');
    });
  });

  describe('assessment columns', () => {
    const traceWithAssessment = makeTrace('tr-000', { assessments: [makeFeedbackAssessment('relevance', 'yes')] });

    test('renders a column per on-page assessment with its value tag', async () => {
      state.pages = { '': { traces: [traceWithAssessment], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('columnheader', { name: 'relevance' })).toBeInTheDocument();
      // 'yes' feedback renders as a "Yes" tag (AssessmentDisplayValue).
      expect(screen.getByText('Yes')).toBeInTheDocument();
    });

    test('offers a single toggle to show/hide all assessment columns at once', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': {
          traces: [
            makeTrace('tr-000', {
              assessments: [makeFeedbackAssessment('relevance', 'yes'), makeFeedbackAssessment('correctness', 'no')],
            }),
          ],
          next_page_token: undefined,
        },
      };
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByRole('columnheader', { name: 'relevance' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'correctness' })).toBeInTheDocument();

      // The assessment group header is itself a checkbox that toggles every assessment column at once
      // (the V3 affordance the V4 table had lost).
      await openDisplaySubmenu(user, /^Columns/);
      const allToggle = await screen.findByRole('menuitemcheckbox', { name: 'Assessments' });
      expect(allToggle).toHaveAttribute('aria-checked', 'true');

      await selectSubmenuItem('menuitemcheckbox', 'Assessments');
      await waitFor(() => expect(screen.queryByRole('columnheader', { name: 'relevance' })).not.toBeInTheDocument());
      expect(screen.queryByRole('columnheader', { name: 'correctness' })).not.toBeInTheDocument();
    }, 20000);

    test('toggling an assessment off hides its column and the choice persists', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = { '': { traces: [traceWithAssessment], next_page_token: undefined } };
      const { unmount } = renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByRole('columnheader', { name: 'relevance' })).toBeInTheDocument();

      await openDisplaySubmenu(user, /^Columns/);
      await selectSubmenuItem('menuitemcheckbox', 'relevance');
      await waitFor(() => expect(screen.queryByRole('columnheader', { name: 'relevance' })).not.toBeInTheDocument());

      // Remount: the opt-out persists per experiment even though the page still has the assessment.
      unmount();
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.queryByRole('columnheader', { name: 'relevance' })).not.toBeInTheDocument();
      // Two full page renders (mount + remount) under parallel jsdom load — bump off the 5s default.
    }, 20000);
  });

  describe('tags column', () => {
    test('is visible by default and shows user tags with a "+N" preview', async () => {
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('columnheader', { name: 'Tags' })).toBeInTheDocument();
      // First tag is shown as a pill; the second is collapsed into "+1". The responsive tags cell
      // renders a hidden measurement copy alongside the visible pill in jsdom, so match on count.
      expect(screen.getAllByText('env: prod').length).toBeGreaterThan(0);
      expect(screen.getAllByText('+1').length).toBeGreaterThan(0);
      // Heavy full-page render is slow under parallel jsdom load; per-test timeout avoids the
      // lint-forbidden global jest.setTimeout.
    }, 20000);

    test('hides internal mlflow.* tags from the preview', async () => {
      // Only an internal tag → nothing user-facing → renders the "-" empty value.
      state.pages = {
        '': {
          traces: [makeTrace('tr-000', { tags: { 'mlflow.trace.sizeBytes': '123' } })],
          next_page_token: undefined,
        },
      };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('columnheader', { name: 'Tags' })).toBeInTheDocument();
      expect(screen.queryByText(/mlflow\.trace\.sizeBytes/)).not.toBeInTheDocument();
    });
  });

  describe('filter by tag', () => {
    test('clicking the first tag pill applies a tags clause and writes the tag URL param', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      await user.click(screen.getByRole('button', { name: 'Filter by tag env: prod' }));

      // The refetch carries the compiled tag clause…
      await waitFor(() => expect(state.searchCalls.some((c) => c.filter?.includes("tags.env = 'prod'"))).toBe(true));
      // …and the filter is persisted in the URL as a repeatable, encoded `tag` param.
      expect(new URLSearchParams(env.lastSearch).getAll('tag')).toContain('env=prod');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('clicking a tag in the overflow hover card applies that tag (not just the first pill)', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      // Hover the tags cell's "+N" activator (the hover-card trigger) to reveal the full tag list,
      // then click the second tag's filter pill — the `team: ml` pill lives only in the overflow list
      // (the inline pill is the first tag, `env: prod`), so finding it proves the hover card opened
      // with its content visible. `fireEvent.click` (not `user.click`) is deliberate: userEvent moves
      // the pointer, which hovers out of the trigger and dismisses the card before the click lands.
      await user.hover(screen.getByRole('button', { name: 'Open trace tr-000 — tags' }));
      fireEvent.click(await screen.findByRole('button', { name: 'Filter by tag team: ml' }));

      await waitFor(() => expect(state.searchCalls.some((c) => c.filter?.includes("tags.team = 'ml'"))).toBe(true));
      expect(new URLSearchParams(env.lastSearch).getAll('tag')).toContain('team=ml');
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('clicking a tag pill filters but does NOT open the trace drawer', async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      state.pages = {
        '': { traces: [makeTaggedTrace('tr-000', { env: 'prod', team: 'ml' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');

      await user.click(screen.getByRole('button', { name: 'Filter by tag env: prod' }));

      // The filter applied (URL param written)…
      await waitFor(() => expect(new URLSearchParams(env.lastSearch).getAll('tag')).toContain('env=prod'));
      // …but the drawer never opened and no traceId was set (stopPropagation kept it off the row click).
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      expect(new URLSearchParams(env.lastSearch).get('traceId')).toBeNull();
    }, 20000); // heavy full-page userEvent render — bump off the flaky 5s default under parallel jsdom load

    test('a deep-linked tag param drives the initial request filter', async () => {
      state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
      renderPage({ initialUrl: `${URL}?tag=env%3Dprod` });
      await findTraceRow('tr-000');

      expect(state.searchCalls[0]?.filter).toContain("tags.env = 'prod'");
    });
  });

  describe('resizable columns', () => {
    test('applies a persisted per-column width on mount', async () => {
      // A width persisted from a prior session (by the resize handle → useLocalStorage) must be
      // read back and applied to the column on the next mount. Seed it via the same scoped/versioned
      // key shape the hook writes, then assert a default-visible header renders at that width. Input
      // is a growing column, so its persisted width becomes the flex-basis (maxWidth is unset so the
      // cap can't block growth).
      const sizesKey = `${TRACE_COLUMN_SIZES_STORAGE_KEY_PREFIX}.${EXPERIMENT_ID}`;
      setLocalStorageItem(sizesKey, COLUMN_SIZES_STORAGE_VERSION, true, { input: 321 });

      renderPage();
      await screen.findByRole('columnheader', { name: 'Input' });
      // Widths are published as `--traces-table-column-<index>` CSS variables on the region and
      // referenced by each header's `flex`. jsdom drops a `flex` shorthand containing `var()`, so we
      // can't read the size off the header's inline style; assert on the published variable instead.
      // 321 is not a default column size, so its presence proves the persisted width flowed through.
      const region = screen.getByRole('region', { name: 'Traces' });
      const publishedWidths = Array.from({ length: region.style.length }, (_, i) => region.style.item(i))
        .filter((prop) => prop.startsWith('--traces-table-column-'))
        .map((prop) => region.style.getPropertyValue(prop));
      expect(publishedWidths).toContain('321px');
    });
  });

  describe('state column', () => {
    test.each([
      { state: 'OK' as const, label: 'OK' },
      { state: 'ERROR' as const, label: 'Error' },
      { state: 'IN_PROGRESS' as const, label: 'In progress' },
    ])('renders the $label badge for a $state trace', async ({ state: traceState, label }) => {
      state.pages = { '': { traces: [makeTrace('tr-000', { state: traceState })], next_page_token: undefined } };
      renderPage();
      await findTraceRow('tr-000');
      expect(screen.getByText(label)).toBeInTheDocument();
    });

    test('renders "-" (no badge) for a STATE_UNSPECIFIED trace', async () => {
      state.pages = {
        '': { traces: [makeTrace('tr-000', { state: 'STATE_UNSPECIFIED' })], next_page_token: undefined },
      };
      renderPage();
      await findTraceRow('tr-000');
      // None of the state labels render for an unspecified state.
      expect(screen.queryByText('OK')).not.toBeInTheDocument();
      expect(screen.queryByText('Error')).not.toBeInTheDocument();
      expect(screen.queryByText('In progress')).not.toBeInTheDocument();
    });
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
      expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
    });

    test('a partial final page disables Next even when the server sends a next_page_token', async () => {
      // Reproduces the reported bug: the long-running backend returns a real token on every non-empty
      // page including a partial final one. 21 rows at the default page size 25 is a short page, so the
      // frontend must treat it as the last page and disable Next — despite the non-empty token.
      state.pages = { '': { traces: makeTraces(21), next_page_token: 'phantom-token' } };
      renderPage();
      await findTraceRow('tr-000');

      expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
      // Heavy full-page render is slow under parallel jsdom load; per-test timeout avoids the
      // lint-forbidden global jest.setTimeout (matches the sibling pagination tests).
    }, 20000);

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
      expect(screen.getByRole('button', { name: 'Next page' })).toBeEnabled();

      await user.click(screen.getByRole('button', { name: 'Next page' }));

      // The distinct end-of-results state renders — NOT the initial "No traces yet" empty state…
      expect(await screen.findByText('No more results')).toBeInTheDocument();
      expect(screen.queryByText('No traces yet')).not.toBeInTheDocument();
      // …the pagination bar is still present (page-size selector lives in it)…
      expect(screen.getByLabelText('Rows per page')).toBeInTheDocument();
      // …Prev is enabled (back to page 1), Next is disabled (the empty page is terminal).
      expect(screen.getByRole('button', { name: 'Previous page' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();

      // Stepping back returns to page 1's rows (served from cache, no refetch needed).
      await user.click(screen.getByRole('button', { name: 'Previous page' }));
      expect(await findTraceRow('p1-000')).toBeInTheDocument();
      // Full-page renders + two paginations are slow under parallel jsdom load; per-test timeout
      // avoids the lint-forbidden global jest.setTimeout.
    }, 30000);

    test('shows the "{n} of {total}" count — current page rows out of the metrics total', async () => {
      // 3 rows on the page; the trace-metrics endpoint reports 42 total.
      state.pages = { '': { traces: makeTraces(3), next_page_token: undefined } };
      env.metricsTotalCount = 42;
      renderPage();
      await findTraceRow('tr-000');

      // The count trails the table (it awaits the separate trace-metrics query, then re-renders the
      // full page), so poll longer than findBy's 1s default — the whole page render is slow under
      // parallel jsdom load (matches the explicit findBy timeouts in TracesV4TraceDrawer.test).
      expect(await screen.findByText('3 of 42', {}, { timeout: 25000 })).toBeInTheDocument();
    }, 30000);
  });

  describe('OSS data path', () => {
    test('OSS searches the MLFLOW_EXPERIMENT location', async () => {
      renderPage();
      await findTraceRow('tr-000');

      const location = state.searchCalls[0]?.locations?.[0];
      // OSS always searches by experiment id; UC-schema locations are a Databricks-only concept.
      expect(location?.type).toBe('MLFLOW_EXPERIMENT');
      expect(location?.mlflow_experiment).toEqual({ experiment_id: EXPERIMENT_ID });
    });
  });
});
