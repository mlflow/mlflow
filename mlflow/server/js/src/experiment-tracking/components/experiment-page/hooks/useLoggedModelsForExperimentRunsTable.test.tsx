import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';

import { act, renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { useLoggedModelsForExperimentRunsTable } from './useLoggedModelsForExperimentRunsTable';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

describe('useLoggedModelsForExperimentRunsTable', () => {
  const server = setupServer();
  let queryClient: QueryClient;

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  const createModel = (modelId: string, runId: string) => ({
    info: { model_id: modelId, source_run_id: runId },
  });

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
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

  afterEach(() => queryClient.clear());

  test('should return logged models for experiment runs', async () => {
    const { result } = renderHook(
      () =>
        useLoggedModelsForExperimentRunsTable({
          experimentIds: ['test-experiment'],
          runIds: ['run-1', 'run-3'],
        }),
      { wrapper },
    );
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

  test('should follow every page, including empty pages, and scope the search to displayed runs', async () => {
    const requests: { page_token?: string; filter?: string }[] = [];
    const firstModel = createModel('model-1', 'run-1');
    const secondModel = createModel('model-2', 'run-1');
    const thirdModel = createModel('model-3', 'run-3');
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', async (req, res, ctx) => {
        const body = await req.json();
        requests.push(body);
        if (!body.page_token) {
          return res(ctx.json({ models: [firstModel], next_page_token: 'second-page' }));
        }
        if (body.page_token === 'second-page') {
          return res(ctx.json({ models: [], next_page_token: 'third-page' }));
        }
        return res(ctx.json({ models: [secondModel, thirdModel] }));
      }),
    );

    const { result } = renderHook(
      () =>
        useLoggedModelsForExperimentRunsTable({
          experimentIds: ['test-experiment'],
          runIds: ['run-1', 'run-3'],
        }),
      { wrapper },
    );

    await waitFor(() => expect(result.current).toEqual({ 'run-1': [firstModel, secondModel], 'run-3': [thirdModel] }));
    expect(requests.map(({ page_token }) => page_token)).toEqual([undefined, 'second-page', 'third-page']);
    expect(requests.every(({ filter }) => filter === "attributes.source_run_id IN ('run-1','run-3')")).toBe(true);
  });

  test('should not fetch models when no runs are displayed', async () => {
    const requestHandler = jest.fn(() => ({ models: [] }));
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', (_req, res, ctx) => res(ctx.json(requestHandler()))),
    );

    renderHook(() => useLoggedModelsForExperimentRunsTable({ experimentIds: ['test-experiment'], runIds: [] }), {
      wrapper,
    });
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(requestHandler).not.toHaveBeenCalled();
  });

  test('should not follow a page token after the hook is disabled', async () => {
    let releaseResponse: () => void = () => {};
    const responseReady = new Promise<void>((resolve) => {
      releaseResponse = resolve;
    });
    const requests: unknown[] = [];
    const firstModel = createModel('model-1', 'run-1');
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', async (req, res, ctx) => {
        requests.push(await req.json());
        await responseReady;
        return res(ctx.json({ models: [firstModel], next_page_token: 'second-page' }));
      }),
    );

    const { result, rerender } = renderHook(
      ({ enabled }) =>
        useLoggedModelsForExperimentRunsTable({ experimentIds: ['test-experiment'], runIds: ['run-1'], enabled }),
      { wrapper, initialProps: { enabled: true } },
    );
    await waitFor(() => expect(requests).toHaveLength(1));
    rerender({ enabled: false });
    await act(async () => releaseResponse());

    await waitFor(() => expect(result.current).toEqual({ 'run-1': [firstModel] }));
    expect(requests).toHaveLength(1);
  });

  test('should preserve loaded models and stop fetching when a subsequent page fails', async () => {
    const consoleError = jest.spyOn(console, 'error').mockImplementation(() => {});
    const requests: (string | undefined)[] = [];
    const firstModel = createModel('model-1', 'run-1');
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', async (req, res, ctx) => {
        const body = await req.json();
        requests.push(body.page_token);
        return body.page_token
          ? res(ctx.status(500), ctx.json({ message: 'Failed to load next page' }))
          : res(ctx.json({ models: [firstModel], next_page_token: 'second-page' }));
      }),
    );

    try {
      const { result } = renderHook(
        () => useLoggedModelsForExperimentRunsTable({ experimentIds: ['test-experiment'], runIds: ['run-1'] }),
        { wrapper },
      );
      await waitFor(() => expect(consoleError).toHaveBeenCalled());
      await waitFor(() => expect(queryClient.isFetching()).toBe(0));
      expect(result.current).toEqual({ 'run-1': [firstModel] });
      expect(requests).toEqual([undefined, 'second-page']);
    } finally {
      consoleError.mockRestore();
    }
  });

  test('should reset the search when the displayed experiment and runs change', async () => {
    const requests: { experiment_ids: string[]; page_token?: string; filter?: string }[] = [];
    server.use(
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', async (req, res, ctx) => {
        const body = await req.json();
        requests.push(body);
        return res(
          ctx.json({ models: [createModel(`model-${body.experiment_ids[0]}`, `run-${body.experiment_ids[0]}`)] }),
        );
      }),
    );

    const { result, rerender } = renderHook(
      ({ experimentId }) =>
        useLoggedModelsForExperimentRunsTable({ experimentIds: [experimentId], runIds: [`run-${experimentId}`] }),
      { wrapper, initialProps: { experimentId: 'first' } },
    );
    await waitFor(() => expect(result.current).toEqual({ 'run-first': [createModel('model-first', 'run-first')] }));
    rerender({ experimentId: 'second' });
    await waitFor(() => expect(result.current).toEqual({ 'run-second': [createModel('model-second', 'run-second')] }));
    expect(requests).toHaveLength(2);
    expect(requests[1]).toEqual(
      expect.objectContaining({ experiment_ids: ['second'], filter: "attributes.source_run_id IN ('run-second')" }),
    );
    expect(requests[1].page_token).toBeUndefined();
  });
});
