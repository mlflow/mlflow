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

  test('should ask the server not to return metric values', async () => {
    let requestBody: Record<string, unknown> | undefined;
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', async (req, res, ctx) => {
        requestBody = await req.json();
        return res(ctx.json({ models: [] }));
      }),
    );

    renderHook(() => useLoggedModelsForExperimentRunsTable({ experimentIds: ['test-experiment'] }), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });

    // The runs table renders each model's name and links to it, and shows no metric
    // values, so asking for them would make the response many times larger for nothing.
    await waitFor(() => {
      expect(requestBody).toEqual(expect.objectContaining({ include_metrics: false }));
    });
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
