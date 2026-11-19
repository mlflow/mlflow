import { beforeAll, beforeEach, describe, expect, test } from '@jest/globals';

import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { useLoggedModelsForExperimentRunsTable } from './useLoggedModelsForExperimentRunsTable';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useSearchLoggedModelsQuery } from '../../../hooks/logged-models/useSearchLoggedModelsQuery';

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

    await waitFor(() => {
      expect(requestBody).toEqual(expect.objectContaining({ include_metrics: false }));
    });
  });

  test('should keep the default metrics request separate from the runs table cache', async () => {
    const requestedValues: boolean[] = [];
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', async (req, res, ctx) => {
        const { include_metrics } = await req.json();
        requestedValues.push(include_metrics);
        return res(
          ctx.json({
            models: [
              {
                info: { model_id: 'model-id', source_run_id: 'run-id' },
                data: { metrics: include_metrics ? [{ key: 'accuracy', value: 0.9 }] : [] },
              },
            ],
          }),
        );
      }),
    );
    const client = new QueryClient();
    const { result } = renderHook(
      () => ({
        table: useLoggedModelsForExperimentRunsTable({ experimentIds: ['test-experiment'] }),
        defaultRequest: useSearchLoggedModelsQuery({ experimentIds: ['test-experiment'] }),
      }),
      { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> },
    );
    await waitFor(() => {
      expect(requestedValues).toEqual(expect.arrayContaining([false, true]));
      expect(result.current.table?.['run-id']?.[0]?.data?.metrics).toEqual([]);
      expect(result.current.defaultRequest.data?.[0].data?.metrics).toEqual([{ key: 'accuracy', value: 0.9 }]);
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
