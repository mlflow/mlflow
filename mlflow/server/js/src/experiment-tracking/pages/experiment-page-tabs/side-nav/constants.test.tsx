import { expect, test } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../../../common/utils/setup-msw';
import { ExperimentKind, ExperimentPageTabName } from '../../../constants';
import { useExperimentPageSideNavConfig } from './constants';

const server = setupServer();

test('updates Gateway-dependent navigation after server-info loads', async () => {
  server.use(
    rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
      return res(
        ctx.json({
          store_type: 'SqlStore',
          workspaces_enabled: false,
          trace_archival_enabled: false,
          multipart_uploads_enabled: false,
          multipart_downloads_enabled: false,
          features_enabled: { gateway: false },
        }),
      );
    }),
  );
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  const { result } = renderHook(
    () => useExperimentPageSideNavConfig({ experimentKind: ExperimentKind.GENAI_DEVELOPMENT }),
    { wrapper },
  );

  await waitFor(() => {
    expect(
      result.current['prompts-versions']?.some(({ tabName }) => tabName === ExperimentPageTabName.Playground),
    ).toBe(false);
  });
  expect(result.current.evaluation?.some(({ tabName }) => tabName === ExperimentPageTabName.Judges)).toBe(false);
});
