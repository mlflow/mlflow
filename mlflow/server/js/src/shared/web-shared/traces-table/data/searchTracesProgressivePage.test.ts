import { describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../test-utils/setup-msw';
import type { ModelTraceSearchLocation } from '../../model-trace-explorer/ModelTrace.types';
import { fetchTracesProgressivePage } from './searchTracesProgressivePage';
import { type TracesQueryIdentity } from './useTracesPageQuery';
import { makeTraces } from '../test-utils/mockTraces';

// Relative `getAjaxUrl` output resolves against jsdom's origin, so match with a `*/` origin wildcard
// (mirrors the sibling useTracesPageQuery MSW test — no MLFLOW_USE_ABSOLUTE_AJAX_URLS needed).
const INITIATE_ENDPOINT = '*/ajax-api/4.0/mlflow/traces/search-progressive';
const OPERATION_ENDPOINT = '*/ajax-api/4.0/mlflow/traces/search-progressive/operations';
// Progressive has no cancel endpoint; a request here would be a bug (long-running-only cancel reused).
const CANCEL_ENDPOINT = '*/ajax-api/2.0/sql/statements/:id/cancel';

const OPERATION_NAME = 'op-token-1';
// A short poll interval keeps the multi-poll test fast without fake timers.
const POLL_INTERVAL_MS = 5;

const IDENTITY: TracesQueryIdentity = {
  locations: [
    { type: 'UC_TABLE_PREFIX', uc_table_prefix: { catalog_name: 'cat', schema_name: 'sch', table_prefix: 'pfx' } },
  ] as ModelTraceSearchLocation[],
  filter: "trace.request ILIKE '%hello%'",
  orderBy: ['timestamp DESC'],
  pageSize: 4,
};

/** A done operation carrying the given traces + token, matching the progressive response shape. */
const doneOp = (traceCount: number, nextPageToken: string | undefined, prefix = 'tr') => ({
  name: OPERATION_NAME,
  done: true,
  response: { trace_infos: makeTraces(traceCount, prefix), next_page_token: nextPageToken },
});

describe('fetchTracesProgressivePage', () => {
  const { server } = setupServer(
    // Default happy path: initiate returns a full page in one round.
    rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json(doneOp(4, undefined)))),
    rest.post(OPERATION_ENDPOINT, (_req, res, ctx) => res(ctx.json(doneOp(4, undefined)))),
  );

  test('accumulates partial batches across rounds until the page is full', async () => {
    // Each initiate returns a pending op; the poll returns a 2-row batch. Two rounds fill pageSize=4.
    // The continuation cursor must be the previous batch's returned token.
    const initiateTokens: (string | undefined)[] = [];
    let round = 0;
    server.use(
      rest.post(INITIATE_ENDPOINT, async (req, res, ctx) => {
        initiateTokens.push(((await req.json()) as { page_token?: string }).page_token);
        return res(ctx.json({ name: OPERATION_NAME, done: false }));
      }),
      rest.post(OPERATION_ENDPOINT, (_req, res, ctx) => {
        round += 1;
        // Round 1 → 2 rows + a token to keep filling; round 2 → 2 more rows + a token (page now full).
        return res(ctx.json(doneOp(2, round === 1 ? 'tok-b' : 'tok-c', round === 1 ? 'a' : 'b')));
      }),
    );

    const result = await fetchTracesProgressivePage(IDENTITY, 'tok-a', undefined, POLL_INTERVAL_MS);

    expect(result.trace_infos).toHaveLength(4);
    expect(result.next_page_token).toBe('tok-c');
    // First round uses the page cursor; second round uses the token returned by the first batch.
    expect(initiateTokens).toEqual(['tok-a', 'tok-b']);
  });

  test('returns a short final page with an undefined token when the search is exhausted', async () => {
    server.use(rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json(doneOp(2, undefined)))));

    const result = await fetchTracesProgressivePage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS);

    expect(result.trace_infos).toHaveLength(2);
    expect(result.next_page_token).toBeUndefined();
  });

  test('returns immediately (no second initiate) when the first batch already fills the page', async () => {
    let initiates = 0;
    server.use(
      rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => {
        initiates += 1;
        return res(ctx.json(doneOp(4, 'more')));
      }),
    );

    const result = await fetchTracesProgressivePage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS);

    expect(initiates).toBe(1);
    expect(result.trace_infos).toHaveLength(4);
    expect(result.next_page_token).toBe('more');
  });

  test('keeps looping past an empty batch that still carries a token', async () => {
    let round = 0;
    server.use(
      rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => {
        round += 1;
        // Round 1 → empty batch + token (keep going); round 2 → the actual rows, exhausted.
        return res(ctx.json(round === 1 ? doneOp(0, 'tok-2') : doneOp(2, undefined)));
      }),
    );

    const result = await fetchTracesProgressivePage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS);

    expect(round).toBe(2);
    expect(result.trace_infos).toHaveLength(2);
    expect(result.next_page_token).toBeUndefined();
  });

  test('continues with the same filter when an empty response omits trace_infos', async () => {
    const filter = "attributes.timestamp_ms < 1785452172606 AND tags.conversation_id = '77'";
    const requestBodies: Array<{ filter?: string; page_token?: string }> = [];
    server.use(
      rest.post(INITIATE_ENDPOINT, async (req, res, ctx) => {
        const body = await req.json<{ filter?: string; page_token?: string }>();
        requestBodies.push({ filter: body.filter, page_token: body.page_token });
        if (requestBodies.length === 1) {
          return res(ctx.json({ name: OPERATION_NAME, done: false }));
        }
        return res(ctx.json(doneOp(2, undefined)));
      }),
      rest.post(OPERATION_ENDPOINT, (_req, res, ctx) =>
        res(
          ctx.json({
            name: OPERATION_NAME,
            done: true,
            response: { next_page_token: 'tok-2' },
          }),
        ),
      ),
    );

    const result = await fetchTracesProgressivePage({ ...IDENTITY, filter }, undefined, undefined, POLL_INTERVAL_MS);

    expect(requestBodies).toEqual([
      { filter, page_token: undefined },
      { filter, page_token: 'tok-2' },
    ]);
    expect(result.trace_infos).toHaveLength(2);
    expect(result.next_page_token).toBeUndefined();
  });

  test('returns an empty page (and does not poll) when initiate completes empty', async () => {
    let polled = false;
    server.use(
      rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json(doneOp(0, undefined)))),
      rest.post(OPERATION_ENDPOINT, (_req, res, ctx) => {
        polled = true;
        return res(ctx.json(doneOp(1, undefined)));
      }),
    );

    const result = await fetchTracesProgressivePage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS);

    expect(result.trace_infos).toEqual([]);
    expect(result.next_page_token).toBeUndefined();
    expect(polled).toBe(false);
  });

  test('sends the progressive payload shape and forwards the page token on a continuation', async () => {
    let body: Record<string, unknown> = {};
    server.use(
      rest.post(INITIATE_ENDPOINT, async (req, res, ctx) => {
        body = (await req.json()) as Record<string, unknown>;
        return res(ctx.json(doneOp(4, undefined)));
      }),
    );

    await fetchTracesProgressivePage(IDENTITY, 'cursor-2', undefined, POLL_INTERVAL_MS);

    expect(body['client']).toBe('UI_WEB');
    expect(body['page_size']).toBe(IDENTITY.pageSize);
    expect(body['location']).toEqual(IDENTITY.locations[0]);
    expect(body).not.toHaveProperty('locations');
    expect(body).not.toHaveProperty('max_results');
    // No `sql_warehouse_id`: warehouse plumbing is Databricks-only and omitted from the OSS payload.
    expect(body).not.toHaveProperty('sql_warehouse_id');
    expect(body['page_token']).toBe('cursor-2');
  });

  test('throws with the operation message when the completed operation carries an error_code', async () => {
    server.use(
      rest.post(INITIATE_ENDPOINT, (_req, res, ctx) =>
        res(
          ctx.json({
            name: OPERATION_NAME,
            done: true,
            response: { error_code: 'PERMISSION_DENIED', message: 'no access' },
          }),
        ),
      ),
    );

    await expect(fetchTracesProgressivePage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS)).rejects.toThrow(
      'no access',
    );
  });

  test('throws when the operation completes with a top-level error', async () => {
    server.use(
      rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json({ name: OPERATION_NAME, done: false }))),
      rest.post(OPERATION_ENDPOINT, (_req, res, ctx) =>
        res(ctx.json({ name: OPERATION_NAME, done: true, error: { error_code: 'INTERNAL_ERROR', message: 'kaboom' } })),
      ),
    );

    await expect(fetchTracesProgressivePage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS)).rejects.toThrow(
      'kaboom',
    );
  });

  test('aborting during polling rejects with AbortError and fires no cancel request', async () => {
    const controller = new AbortController();
    let cancelRequested = false;
    // Deterministic hand-off (no timers): the poll handler signals it has been reached.
    let onFirstPoll: () => void;
    const firstPollSeen = new Promise<void>((resolve) => {
      onFirstPoll = resolve;
    });
    server.use(
      rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json({ name: OPERATION_NAME, done: false }))),
      // Never completes, so the search stays in the poll loop until aborted.
      rest.post(OPERATION_ENDPOINT, (_req, res, ctx) => {
        onFirstPoll();
        return res(ctx.json({ name: OPERATION_NAME, done: false }));
      }),
      rest.post(CANCEL_ENDPOINT, (_req, res, ctx) => {
        cancelRequested = true;
        return res(ctx.json({}));
      }),
    );

    const promise = fetchTracesProgressivePage(IDENTITY, undefined, controller.signal, POLL_INTERVAL_MS);
    // Abort only once the loop has actually started polling — no wall-clock race.
    await firstPollSeen;
    controller.abort();

    await expect(promise).rejects.toThrow('Aborted');
    // The deliberate contrast with long-running: progressive never issues a cancel request.
    expect(cancelRequested).toBe(false);
  });

  test('rejects after exceeding the max round bound when the server never fills the page', async () => {
    // Always returns a token with zero rows → the page never fills and the search never exhausts.
    server.use(rest.post(INITIATE_ENDPOINT, (_req, res, ctx) => res(ctx.json(doneOp(0, 'endless')))));

    await expect(fetchTracesProgressivePage(IDENTITY, undefined, undefined, POLL_INTERVAL_MS)).rejects.toThrow(
      'exceeded the maximum number of rounds',
    );
  });
});
