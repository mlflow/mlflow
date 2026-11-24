import { beforeEach, expect, jest, test } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { shouldEnableExperimentIssuesTab } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { setupServer } from '../../../../common/utils/setup-msw';
import { ExperimentKind, ExperimentPageTabName } from '../../../constants';
import { useServerInfo } from '../../../hooks/useServerInfo';
import { useExperimentPageSideNavConfig } from './constants';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableExperimentIssuesTab: jest.fn(),
}));

const server = setupServer();

beforeEach(() => {
  jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(false);
});

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

test.each([
  ['SqlStore', true],
  ['SqlStore', false],
  ['FileStore', true],
  ['FileStore', false],
  [null, true],
  [null, false],
])(
  'shows Issues only for SQL-backed GenAI experiments with the flag enabled (store: %s, flag: %s)',
  async (storeType, enabled) => {
    jest.mocked(shouldEnableExperimentIssuesTab).mockReturnValue(enabled);
    server.use(
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => res(ctx.json({ store_type: storeType }))),
    );
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
    const { result } = renderHook(
      () => ({
        server: useServerInfo(),
        genai: useExperimentPageSideNavConfig({ experimentKind: ExperimentKind.GENAI_DEVELOPMENT }),
        custom: useExperimentPageSideNavConfig({ experimentKind: ExperimentKind.CUSTOM_MODEL_DEVELOPMENT }),
      }),
      { wrapper },
    );
    await waitFor(() => expect(result.current.server.isLoading).toBe(false));
    expect(result.current.genai.observability?.map(({ tabName }) => tabName)).toEqual([
      ExperimentPageTabName.Traces,
      ...(enabled && storeType === 'SqlStore' ? [ExperimentPageTabName.Issues] : []),
      ExperimentPageTabName.Dashboard,
      ExperimentPageTabName.ChatSessions,
    ]);
    expect(
      Object.values(result.current.custom)
        .flat()
        .some(({ tabName }) => tabName === ExperimentPageTabName.Issues),
    ).toBe(false);
  },
);
