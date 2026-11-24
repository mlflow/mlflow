import { type ReactNode, useState } from 'react';
import { beforeEach, describe, expect, it } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ModelTraceSpan } from '@databricks/web-shared/model-trace-explorer';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { setupServer } from '@databricks/web-shared/test-utils/setup-msw';
import { rest } from 'msw';
import { makeTrace } from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { createComparisonRootSpan } from './evalRunComparisonTestUtils';
import {
  useEvalRunComparisonTraceInputs,
  type EvalRunComparisonTraceInputsParams,
} from './useEvalRunComparisonTraceInputs';

const InputsTestWrapper = ({ children }: { children?: ReactNode }) => {
  // Isolate comparison caches across tests.
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }));
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
};

const defaultParams: EvalRunComparisonTraceInputsParams = {
  experimentId: 'exp',
  traceIds: ['trace-id'],
  enabled: true,
};
const renderInputs = (overrides: Partial<EvalRunComparisonTraceInputsParams> = {}) =>
  renderHook((props: EvalRunComparisonTraceInputsParams) => useEvalRunComparisonTraceInputs(props), {
    initialProps: { ...defaultParams, ...overrides },
    wrapper: InputsTestWrapper,
  });

describe('useEvalRunComparisonTraceInputs', () => {
  const { server, waitForPendingRequests } = setupServer();
  const endpoint = '*/ajax-api/3.0/mlflow/traces/batchGet';
  beforeEach(() => {
    server.use(
      rest.get(endpoint, (req, res, ctx) => {
        const ids = req.url.searchParams.getAll('trace_ids');
        if (ids.length > 10) {
          return res(
            ctx.status(400),
            ctx.json({ error_code: 'INVALID_PARAMETER_VALUE', message: 'Too many trace IDs' }),
          );
        }
        return res(
          ctx.json({
            traces: ids.map((id) => ({
              trace_info: makeTrace(id),
              spans: [createComparisonRootSpan(`${req.url.searchParams.getAll('experiment_ids').join(',')}:${id}`)],
            })),
          }),
        );
      }),
    );
  });

  it.each<{ label: string; spans: ModelTraceSpan[]; expected?: string }>([
    {
      label: 'MLflow root input before other conventions and child spans',
      spans: [
        createComparisonRootSpan('child', { parent_span_id: '0123456789abcdef' }),
        createComparisonRootSpan('  {"full": true}  ', {
          attributes: {
            'mlflow.spanInputs': '  {"full": true}  ',
            'input.value': 'other',
            'traceloop.entity.input': 'last',
          },
        }),
      ],
      expected: '  {"full": true}  ',
    },
    { label: 'empty recorded input', spans: [createComparisonRootSpan('')], expected: '' },
    {
      label: 'OpenInference input',
      spans: [
        createComparisonRootSpan('', { attributes: { 'input.value': 'open', 'traceloop.entity.input': 'last' } }),
      ],
      expected: 'open',
    },
    {
      label: 'Traceloop input',
      spans: [createComparisonRootSpan('', { attributes: { 'traceloop.entity.input': 'loop' } })],
      expected: 'loop',
    },
    {
      label: 'OTEL attribute array',
      spans: [
        {
          ...createComparisonRootSpan(''),
          kind: 'SPAN_KIND_INTERNAL',
          attributes: [{ key: 'mlflow.spanInputs', value: { string_value: 'otel' } }],
        },
      ],
      expected: 'otel',
    },
    {
      label: 'structured OTEL input',
      spans: [
        {
          ...createComparisonRootSpan(''),
          attributes: [
            {
              key: 'mlflow.spanInputs',
              value: { kvlist_value: { values: [{ key: 'query', value: { string_value: 'hello' } }] } },
            },
          ],
        },
      ],
      expected: '{"query":"hello"}',
    },
    { label: 'missing root', spans: [createComparisonRootSpan('child', { parent_span_id: '0123456789abcdef' })] },
    { label: 'missing input', spans: [createComparisonRootSpan('', { attributes: {} })] },
    { label: 'non-string input', spans: [createComparisonRootSpan('', { attributes: { 'mlflow.spanInputs': 42 } })] },
  ])('handles $label', async ({ spans, expected }) => {
    server.use(
      rest.get(endpoint, (_req, res, ctx) => res(ctx.json({ traces: [{ trace_info: makeTrace('trace-id'), spans }] }))),
    );
    const { result } = renderInputs();
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.inputsByTraceId).toEqual(
      expected === undefined ? new Map() : new Map([['trace-id', expected]]),
    );
  });

  it.each([
    { label: 'disabled', overrides: { enabled: false } },
    { label: 'empty IDs', overrides: { traceIds: [] } },
    { label: 'missing experiment', overrides: { experimentId: '' } },
  ])('does not fetch or remain loading with $label', async ({ overrides }) => {
    const requests: string[] = [];
    server.use(
      rest.get(endpoint, (req, res, ctx) => {
        requests.push(req.url.href);
        return res(ctx.json({ traces: [] }));
      }),
    );
    const { result } = renderInputs(overrides);
    await act(async () => {
      await result.current.refetch();
    });
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetching).toBe(false);
    expect(requests).toEqual([]);
  });

  it.each([10, 11, 25])('loads all recorded inputs for %i traces within the backend batch limit', async (count) => {
    const ids = Array.from({ length: count }, (_, index) => `id-${String(index).padStart(3, '0')}`);
    const { result } = renderInputs({ traceIds: ids });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.inputsByTraceId).toEqual(new Map(ids.map((id) => [id, `exp:${id}`])));
  });

  it('deduplicates IDs and limits batches and concurrent requests', async () => {
    const ids = Array.from({ length: 31 }, (_, index) => `id-${String(index).padStart(3, '0')}`);
    const requests: string[][] = [];
    let inFlight = 0;
    let peak = 0;
    let release = () => {};
    const ready = new Promise<void>((resolve) => {
      release = resolve;
    });
    server.use(
      rest.get(endpoint, async (req, res, ctx) => {
        const batch = req.url.searchParams.getAll('trace_ids');
        requests.push(batch);
        if (batch.length > 10) {
          return res(
            ctx.status(400),
            ctx.json({ error_code: 'INVALID_PARAMETER_VALUE', message: 'Too many trace IDs' }),
          );
        }
        inFlight += 1;
        peak = Math.max(peak, inFlight);
        await ready;
        inFlight -= 1;
        return res(
          ctx.json({
            traces: batch.map((id) => ({ trace_info: makeTrace(id), spans: [createComparisonRootSpan(id)] })),
          }),
        );
      }),
    );
    const { result } = renderInputs({ traceIds: [...ids, ...ids] });
    try {
      await waitFor(() => expect(requests).toHaveLength(2));
      expect(result.current.isLoading).toBe(true);
      expect(requests.map((batch) => batch.length)).toEqual([10, 10]);
    } finally {
      release();
    }
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.inputsByTraceId).toEqual(new Map(ids.map((id) => [id, id])));
    expect(requests.flat()).toEqual(ids);
    expect(peak).toBe(2);
  });

  it('retains successful batches when another batch fails or omits a trace', async () => {
    const ids = Array.from({ length: 11 }, (_, index) => `id-${String(index).padStart(3, '0')}`);
    server.use(
      rest.get(endpoint, (req, res, ctx) => {
        const batch = req.url.searchParams.getAll('trace_ids');
        return batch.includes('id-000')
          ? res(ctx.status(500), ctx.json({ message: 'Unavailable' }))
          : res(ctx.json({ traces: [{ trace_info: makeTrace('id-010'), spans: [createComparisonRootSpan('kept')] }] }));
      }),
    );
    const { result } = renderInputs({ traceIds: ids });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.inputsByTraceId).toEqual(new Map([['id-010', 'kept']]));
  });

  it.each([
    { label: 'experiment', next: { ...defaultParams, experimentId: 'other' }, expected: 'other:trace-id' },
    { label: 'trace set', next: { ...defaultParams, traceIds: ['another'] }, expected: 'exp:another' },
  ])('isolates input data after a $label change', async ({ next, expected }) => {
    const { result, rerender } = renderInputs();
    await waitFor(() => expect(result.current.inputsByTraceId.get('trace-id')).toBe('exp:trace-id'));
    rerender(next);
    expect(result.current.isLoading).toBe(true);
    expect(result.current.inputsByTraceId.size).toBe(0);
    await waitFor(() => expect(Array.from(result.current.inputsByTraceId.values())).toEqual([expected]));
  });

  it('reuses cached inputs for reordered IDs and reloads them on explicit refresh', async () => {
    const { result, rerender } = renderInputs({ traceIds: ['a', 'b'] });
    await waitFor(() => expect(result.current.inputsByTraceId.size).toBe(2));
    const requests: string[][] = [];
    server.use(
      rest.get(endpoint, (req, res, ctx) => {
        const ids = req.url.searchParams.getAll('trace_ids');
        requests.push(ids);
        return res(
          ctx.json({
            traces: ids.map((id) => ({ trace_info: makeTrace(id), spans: [createComparisonRootSpan('refreshed')] })),
          }),
        );
      }),
    );
    rerender({ ...defaultParams, traceIds: ['b', 'a'] });
    expect(result.current.inputsByTraceId.get('a')).toBe('exp:a');
    expect(result.current.isFetching).toBe(false);
    expect(requests).toEqual([]);
    await act(async () => {
      await result.current.refetch();
    });
    await waitFor(() =>
      expect(result.current.inputsByTraceId).toEqual(
        new Map([
          ['a', 'refreshed'],
          ['b', 'refreshed'],
        ]),
      ),
    );
    expect(requests).toEqual([['a', 'b']]);
  });

  it('cancels obsolete loading without publishing old inputs or starting queued batches', async () => {
    const oldIds = Array.from({ length: 21 }, (_, index) => `old-${index}`);
    const requests: string[][] = [];
    let release = () => {};
    const ready = new Promise<void>((resolve) => {
      release = resolve;
    });
    server.use(
      rest.get(endpoint, async (req, res, ctx) => {
        const ids = req.url.searchParams.getAll('trace_ids');
        requests.push(ids);
        if (ids.some((id) => id.startsWith('old-'))) {
          await ready;
        }
        return res(
          ctx.json({ traces: ids.map((id) => ({ trace_info: makeTrace(id), spans: [createComparisonRootSpan(id)] })) }),
        );
      }),
    );
    const { result, rerender } = renderInputs({ traceIds: oldIds });
    try {
      await waitFor(() => expect(requests).toHaveLength(2));
      rerender({ ...defaultParams, traceIds: ['new'] });
      await waitFor(() => expect(result.current.inputsByTraceId).toEqual(new Map([['new', 'new']])));
    } finally {
      release();
    }
    await waitForPendingRequests();
    expect(requests).toHaveLength(3);
    expect(result.current.inputsByTraceId).toEqual(new Map([['new', 'new']]));
  });
});
