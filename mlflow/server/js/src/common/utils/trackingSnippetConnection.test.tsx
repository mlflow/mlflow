import { afterEach, describe, expect, it } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import type { ReactNode } from 'react';

import { QueryClient, QueryClientProvider } from './reactQueryHooks';
import { setupServer } from './setup-msw';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { getTrackingServerUri, useTrackingSnippetConnection } from './trackingSnippetConnection';

const server = setupServer();

const renderConnection = (queryClient = new QueryClient()) =>
  renderHook(() => useTrackingSnippetConnection(), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });

describe('tracking snippet connection', () => {
  afterEach(() => {
    document.querySelector('base')?.remove();
    delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
    setActiveWorkspace(null);
  });

  it('uses the same base path as the relative server-info request', () => {
    const base = document.createElement('base');
    base.href = 'https://example.com/mlflow/';
    document.head.appendChild(base);
    expect(getTrackingServerUri()).toBe('https://example.com/mlflow');
  });

  it('uses the origin when API requests use absolute paths', () => {
    const base = document.createElement('base');
    base.href = 'https://example.com/mlflow/';
    document.head.appendChild(base);
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    expect(getTrackingServerUri()).toBe('https://example.com');
  });

  it('includes the active workspace and updates when it changes', () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(['serverInfo'], { tracking_auth: '' });
    const { result } = renderConnection(queryClient);
    expect(result.current.workspace).toBeNull();
    act(() => setActiveWorkspace('team-a'));
    expect(result.current.workspace).toBe('team-a');
  });

  it('adds tracking auth when server-info resolves after the first render', async () => {
    let releaseResponse: () => void = () => {};
    const responseReady = new Promise<void>((resolve) => {
      releaseResponse = resolve;
    });
    server.use(
      rest.get('/ajax-api/3.0/mlflow/server-info', async (_req, res, ctx) => {
        await responseReady;
        return res(ctx.json({ tracking_auth: 'kubernetes' }));
      }),
    );

    const { result } = renderConnection();
    expect(result.current.trackingAuth).toBeUndefined();
    releaseResponse();
    await waitFor(() => expect(result.current.trackingAuth).toBe('kubernetes'));
  });
});
