import { describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../test-utils/setup-msw';
import type { ModelTraceSearchLocation } from '../../model-trace-explorer/ModelTrace.types';
import { fetchTracesLongRunningPage } from './searchTracesLongRunningPage';
import { type TracesQueryIdentity } from './useTracesPageQuery';
import { makeTraces } from '../test-utils/mockTraces';

// Relative `getAjaxUrl` output resolves against jsdom's origin, so match with a `*/` origin wildcard
// (mirrors the sibling useTracesPageQuery MSW test — no MLFLOW_USE_ABSOLUTE_AJAX_URLS needed).
const INITIATE_ENDPOINT = '*/ajax-api/4.0/mlflow/traces/search-long-running';
const OPERATION_ENDPOINT = '*/ajax-api/4.0/mlflow/traces/search/operations/:name';
const CANCEL_ENDPOINT = '*/ajax-api/2.0/sql/statements/:id/cancel';

const OPERATION_NAME = 'stmt-123';
// A short poll interval keeps the multi-poll test fast without fake timers.
const POLL_INTERVAL_MS = 5;

const IDENTITY: TracesQueryIdentity = {
  locations: [
    { type: 'UC_SCHEMA', uc_schema: { catalog_name: 'cat', schema_name: 'sch' } },
  ] as ModelTraceSearchLocation[],
  filter: "trace.request ILIKE '%hello%'",
  orderBy: ['timestamp DESC'],
  sqlWarehouseId: 'wh-1',
  pageSize: 25,
};

describe('fetchTracesLongRunningPage', () => {
  const { server, waitForPendingRequests } = setupServer(
    // Default happy path: initiate returns a pending op, the first poll completes with two traces.
    rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json({ name: OPERATION_NAME, done: false }))),
    rest.get(OPERATION_ENDPOINT, (_req, res, ctx) =>
      res(
        ctx.json({
          name: OPERATION_NAME,
          done: true,
          response: { trace_infos: makeTraces(2), next_page_token: 'next' },
        }),
      ),
    ),
  );

  test('initiates then polls until done, returning traces and the next-page token', async () => {
    // Not-done on the first poll, done on the second — exercises the poll loop across the interval.
    let polls = 0;
    server.use(
      rest.get(OPERATION_ENDPOINT, (_req, res, ctx) => {
        polls += 1;
        if (polls < 2) {
          return res(ctx.json({ name: OPERATION_NAME, done: false }));
        }
        return res(
          ctx.json({
            name: OPERATION_NAME,
            done: true,
            response: { trace_infos: makeTraces(2), next_page_token: 'next' },
          }),
        );
      }),
    );

    const result = await fetchTracesLongRunningPage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS);

    expect(polls).toBe(2);
    expect(result.trace_infos?.map((t) => t.trace_id)).toEqual(['tr-000', 'tr-001']);
    expect(result.next_page_token).toBe('next');
  });

  test('forwards the page token on the initiate request', async () => {
    let initiatedToken: string | undefined;
    server.use(
      rest.post(INITIATE_ENDPOINT, async (req, res, ctx) => {
        initiatedToken = ((await req.json()) as { page_token?: string }).page_token;
        return res(ctx.json({ name: OPERATION_NAME, done: false }));
      }),
    );

    await fetchTracesLongRunningPage(IDENTITY, 'cursor-2', undefined, POLL_INTERVAL_MS);

    expect(initiatedToken).toBe('cursor-2');
  });

  test('throws with the operation message when the completed operation carries an error_code', async () => {
    server.use(
      rest.get(OPERATION_ENDPOINT, (_req, res, ctx) =>
        res(
          ctx.json({
            name: OPERATION_NAME,
            done: true,
            response: { error_code: 'PERMISSION_DENIED', message: 'no access' },
          }),
        ),
      ),
    );

    await expect(fetchTracesLongRunningPage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS)).rejects.toThrow(
      'no access',
    );
  });

  test('throws when the operation completes with a top-level error', async () => {
    server.use(
      rest.get(OPERATION_ENDPOINT, (_req, res, ctx) =>
        res(
          ctx.json({ name: OPERATION_NAME, done: true, error: { error_code: 'CANCELLED', message: 'was cancelled' } }),
        ),
      ),
    );

    await expect(fetchTracesLongRunningPage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS)).rejects.toThrow(
      'was cancelled',
    );
  });

  test('returns an empty page when the operation completes with no trace_infos', async () => {
    server.use(
      rest.get(OPERATION_ENDPOINT, (_req, res, ctx) =>
        res(ctx.json({ name: OPERATION_NAME, done: true, response: { trace_infos: [] } })),
      ),
    );

    const result = await fetchTracesLongRunningPage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS);

    expect(result.trace_infos).toEqual([]);
    expect(result.next_page_token).toBeUndefined();
  });

  test('returns an empty page (and does not poll) when initiate returns no operation name', async () => {
    let polled = false;
    server.use(
      rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json({}))),
      rest.get(OPERATION_ENDPOINT, (_req, res, ctx) => {
        polled = true;
        return res(ctx.json({ name: OPERATION_NAME, done: true, response: { trace_infos: makeTraces(1) } }));
      }),
    );

    const result = await fetchTracesLongRunningPage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS);

    expect(result.trace_infos).toEqual([]);
    expect(polled).toBe(false);
  });

  test('aborting during polling cancels the operation and rejects with AbortError', async () => {
    const controller = new AbortController();
    let cancelledStatementId: string | undefined;
    // Deterministic hand-offs (no timers): the poll handler signals it has been reached, and the
    // cancel handler signals it has received the fire-and-forget cancel.
    let onFirstPoll: () => void;
    const firstPollSeen = new Promise<void>((resolve) => {
      onFirstPoll = resolve;
    });
    let onCancel: () => void;
    const cancelSeen = new Promise<void>((resolve) => {
      onCancel = resolve;
    });
    server.use(
      // Never completes, so the search stays in the poll loop until aborted.
      rest.get(OPERATION_ENDPOINT, (_req, res, ctx) => {
        onFirstPoll();
        return res(ctx.json({ name: OPERATION_NAME, done: false }));
      }),
      rest.post(CANCEL_ENDPOINT, (req, res, ctx) => {
        cancelledStatementId = String(req.params['id']);
        onCancel();
        return res(ctx.json({}));
      }),
    );

    const promise = fetchTracesLongRunningPage(IDENTITY, undefined, controller.signal, POLL_INTERVAL_MS);
    // Abort only once the loop has actually started polling — no wall-clock race.
    await firstPollSeen;
    controller.abort();

    await expect(promise).rejects.toThrow('Aborted');
    // Abort fires the fire-and-forget cancel synchronously; await its arrival deterministically.
    await cancelSeen;
    expect(cancelledStatementId).toBe(OPERATION_NAME);
    // Drain the fire-and-forget cancel so the strict MSW wrapper doesn't flag it as pending at teardown.
    await waitForPendingRequests();
  });
});
