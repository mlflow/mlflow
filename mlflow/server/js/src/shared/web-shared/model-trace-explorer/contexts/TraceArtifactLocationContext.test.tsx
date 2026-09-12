import { describe, jest, beforeAll, beforeEach, afterAll, afterEach, it, expect } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

import { TraceArtifactLocationContextProvider } from './TraceArtifactLocationContext';
import type { ModelTrace } from '../ModelTrace.types';
import { useAttachmentUrl } from '../attachment-utils';
import { useTraceAttachment } from '../hooks/useTraceAttachment';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

const mockRequestedUrls: string[] = [];

// `useTraceAttachment` fetches via the common FetchUtils helper, which the global test
// setup blocks; record the URL it would have requested instead.
jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchOrFail: jest.fn(async (url: string) => {
    mockRequestedUrls.push(url);
    return { blob: async () => new Blob(['data']) };
  }),
}));

const TRACE_ID = 'tr-abc';
const ATTACHMENT_ID = 'att-1';
const PROXY_ROOT = `${window.location.origin}/api/2.0/mlflow-artifacts/artifacts/1/traces/${TRACE_ID}/artifacts`;
const PROXY_URL = `${PROXY_ROOT}/attachments/${ATTACHMENT_ID}`;

const makeTrace = (artifactLocation?: string, traceId = TRACE_ID): ModelTrace =>
  ({
    data: { spans: [] },
    info: {
      request_id: traceId,
      tags: artifactLocation ? [{ key: 'mlflow.artifactLocation', value: artifactLocation }] : [],
    },
  }) as unknown as ModelTrace;

describe('TraceArtifactLocationContext', () => {
  const server = setupServer(
    rest.get('*', (req, res, ctx) => {
      mockRequestedUrls.push(req.url.toString());
      return res(ctx.status(200), ctx.body('data'));
    }),
  );

  beforeAll(() => server.listen());
  afterAll(() => server.close());
  beforeEach(() => {
    mockRequestedUrls.length = 0;
    global.URL.createObjectURL = jest.fn(() => 'blob:mock');
    global.URL.revokeObjectURL = jest.fn();
  });
  afterEach(() => server.resetHandlers());

  const makeWrapper =
    (modelTrace: ModelTrace) =>
    ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <TraceArtifactLocationContextProvider modelTrace={modelTrace}>{children}</TraceArtifactLocationContextProvider>
      </QueryClientProvider>
    );

  it('fetches attachments from the stored proxy URI', async () => {
    renderHook(() => useTraceAttachment({ traceId: TRACE_ID, attachmentId: ATTACHMENT_ID, contentType: 'image/png' }), {
      wrapper: makeWrapper(makeTrace(PROXY_ROOT)),
    });

    await waitFor(() => expect(mockRequestedUrls).toHaveLength(1));
    expect(mockRequestedUrls[0]).toBe(PROXY_URL);
  });

  it('falls back to the tracking endpoint when the trace has no proxy artifact location', async () => {
    renderHook(() => useTraceAttachment({ traceId: TRACE_ID, attachmentId: ATTACHMENT_ID, contentType: 'image/png' }), {
      wrapper: makeWrapper(makeTrace('mlflow-artifacts:/1/traces/tr-abc/artifacts')),
    });

    await waitFor(() => expect(mockRequestedUrls).toHaveLength(1));
    expect(mockRequestedUrls[0]).toContain('get-trace-artifact');
  });

  it('does not apply one trace\u2019s artifact location to another trace\u2019s attachment', async () => {
    renderHook(
      () => useTraceAttachment({ traceId: 'tr-other', attachmentId: ATTACHMENT_ID, contentType: 'image/png' }),
      { wrapper: makeWrapper(makeTrace(PROXY_ROOT)) },
    );

    await waitFor(() => expect(mockRequestedUrls).toHaveLength(1));
    expect(mockRequestedUrls[0]).toContain('get-trace-artifact');
  });

  it('routes useAttachmentUrl through the stored proxy URI', async () => {
    renderHook(
      () => useAttachmentUrl(`mlflow-attachment://${ATTACHMENT_ID}?content_type=image%2Fpng&trace_id=${TRACE_ID}`),
      { wrapper: makeWrapper(makeTrace(PROXY_ROOT)) },
    );

    await waitFor(() => expect(mockRequestedUrls).toHaveLength(1));
    expect(mockRequestedUrls[0]).toBe(PROXY_URL);
  });

  it('falls back to the tracking endpoint outside the provider', async () => {
    renderHook(() => useTraceAttachment({ traceId: TRACE_ID, attachmentId: ATTACHMENT_ID, contentType: 'image/png' }), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          {children}
        </QueryClientProvider>
      ),
    });

    await waitFor(() => expect(mockRequestedUrls).toHaveLength(1));
    expect(mockRequestedUrls[0]).toContain('get-trace-artifact');
  });
});
