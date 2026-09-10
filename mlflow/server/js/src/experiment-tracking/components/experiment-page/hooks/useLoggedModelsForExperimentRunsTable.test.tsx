import { beforeAll, beforeEach, describe, expect, test } from '@jest/globals';

import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { useLoggedModelsForExperimentRunsTable } from './useLoggedModelsForExperimentRunsTable';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

describe('useLoggedModelsForExperimentRunsTable', () => {
  const server = setupServer();

  beforeAll(() => server.listen());

  beforeEach(() => {
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) =>
        res(
          ctx.json({
            models: [
              {
                info: {
                  source_run_id: 'run-1',
                  name: 'test-logged-model-1',
                  experiment_id: 'test-experiment',
                  model_id: 'model-id-1',
                },
              },
              {
                info: {
                  source_run_id: 'run-1',
                  name: 'test-logged-model-2',
                  experiment_id: 'test-experiment',
                  model_id: 'model-id-2',
                },
              },
              {
                info: {
                  source_run_id: 'run-3',
                  name: 'test-logged-model-3',
                  experiment_id: 'test-experiment',
                  model_id: 'model-id-3',
                },
              },
            ],
          }),
        ),
      ),
    );
  });

  test('should fetch all pages so runs beyond the first page still get their models', async () => {
    // SearchLoggedModels returns a bounded page (50 by default), so a run's models can sit on any
    // page. Serve two pages and assert the hook drains both instead of stopping at the first.
    const requestedPageTokens: (string | undefined)[] = [];
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) => {
        const pageToken = (req.body as { page_token?: string } | undefined)?.page_token;
        requestedPageTokens.push(pageToken);
        if (!pageToken) {
          return res(
            ctx.json({
              models: [{ info: { source_run_id: 'run-on-page-1', model_id: 'model-id-1' } }],
              next_page_token: 'page-2-token',
            }),
          );
        }
        return res(ctx.json({ models: [{ info: { source_run_id: 'run-on-page-2', model_id: 'model-id-2' } }] }));
      }),
    );

    const { result } = renderHook(() => useLoggedModelsForExperimentRunsTable({ experimentIds: ['test-experiment'] }), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });

    await waitFor(() => {
      expect(result.current).toEqual({
        'run-on-page-1': [{ info: expect.objectContaining({ model_id: 'model-id-1' }) }],
        'run-on-page-2': [{ info: expect.objectContaining({ model_id: 'model-id-2' }) }],
      });
    });

    expect(requestedPageTokens).toEqual([undefined, 'page-2-token']);
  });

  test('should return logged models for experiment runs', async () => {
    const { result } = renderHook(() => useLoggedModelsForExperimentRunsTable({ experimentIds: ['test-experiment'] }), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });
    await waitFor(() => {
      expect(result.current).toEqual({
        'run-1': [
          {
            info: expect.objectContaining({
              source_run_id: 'run-1',
              name: 'test-logged-model-1',
              experiment_id: 'test-experiment',
              model_id: 'model-id-1',
            }),
          },
          {
            info: expect.objectContaining({
              source_run_id: 'run-1',
              name: 'test-logged-model-2',
              experiment_id: 'test-experiment',
              model_id: 'model-id-2',
            }),
          },
        ],
        'run-3': [
          {
            info: expect.objectContaining({
              source_run_id: 'run-3',
              name: 'test-logged-model-3',
              experiment_id: 'test-experiment',
              model_id: 'model-id-3',
            }),
          },
        ],
      });
    });
  });
});
