import { describe, beforeAll, beforeEach, test, expect } from '@jest/globals';

import { act, renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { useExperimentEvaluationRunsData } from './useExperimentEvaluationRunsData';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

describe('useExperimentEvaluationRunsData', () => {
  const server = setupServer();

  beforeAll(() => server.listen());

  beforeEach(() => {
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) =>
        res(
          ctx.json({
            runs: [
              {
                info: {
                  run_uuid: 'run-1',
                  name: 'test-logged-model-1',
                  experiment_id: 'test-experiment',
                },
                outputs: {
                  model_outputs: [
                    {
                      model_id: 'm-1',
                    },
                  ],
                },
              },
              {
                info: {
                  run_uuid: 'run-2',
                  name: 'test-logged-model-1',
                  experiment_id: 'test-experiment',
                },
              },
            ],
          }),
        ),
      ),
    );
  });

  test('should separate runs with and without model outputs', async () => {
    const { result } = renderHook(
      () => useExperimentEvaluationRunsData({ experimentId: 'test-experiment', enabled: true, filter: '' }),
      {
        wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
      },
    );
    await waitFor(() => {
      // expect only the run with no model output
      expect(result.current.data).toHaveLength(1);
      expect(result.current.data?.[0]).toEqual(
        expect.objectContaining({
          info: expect.objectContaining({
            run_uuid: 'run-2',
            name: 'test-logged-model-1',
            experiment_id: 'test-experiment',
          }),
        }),
      );
    });

    expect(result.current.trainingRuns).toHaveLength(1);
    expect(result.current.trainingRuns?.[0]).toEqual(
      expect.objectContaining({
        info: expect.objectContaining({
          run_uuid: 'run-1',
          name: 'test-logged-model-1',
          experiment_id: 'test-experiment',
        }),
      }),
    );
  });

  test('sends the filter built at request time when resolveFilter is given, and keeps the key stable', async () => {
    const sentFilters: string[] = [];
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', async (req, res, ctx) => {
        sentFilters.push((await req.json()).filter);
        return res(ctx.json({ runs: [] }));
      }),
    );
    let call = 0;
    const resolveFilter = () => `attributes.start_time > ${++call}`;
    const queryClient = new QueryClient();
    const { result } = renderHook(
      () =>
        useExperimentEvaluationRunsData({
          experimentId: 'test-experiment',
          enabled: true,
          filter: 'attributes.start_time > 0',
          resolveFilter,
        }),
      { wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider> },
    );
    await waitFor(() => expect(sentFilters).toEqual(['attributes.start_time > 1']));

    await act(async () => {
      await result.current.refetch();
    });

    expect(sentFilters).toEqual(['attributes.start_time > 1', 'attributes.start_time > 2']);
    expect(queryClient.getQueryCache().findAll({ queryKey: ['SEARCH_RUNS'] })).toHaveLength(1);
  });

  test("uses the first page's resolved filter for later pages, and resolves again on refetch", async () => {
    const sent: { filter: string; pageToken?: string }[] = [];
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', async (req, res, ctx) => {
        const body = await req.json();
        sent.push({ filter: body.filter, pageToken: body.page_token });
        return res(ctx.json({ runs: [], ...(body.page_token ? {} : { next_page_token: 'page-2' }) }));
      }),
    );
    let call = 0;
    const resolveFilter = () => `attributes.start_time > ${++call}`;
    const { result } = renderHook(
      () =>
        useExperimentEvaluationRunsData({
          experimentId: 'test-experiment',
          enabled: true,
          filter: 'attributes.start_time > 0',
          resolveFilter,
        }),
      { wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider> },
    );
    await waitFor(() => expect(result.current.hasNextPage).toBe(true));

    await act(async () => {
      await result.current.fetchNextPage();
    });
    expect(sent).toEqual([
      { filter: 'attributes.start_time > 1', pageToken: undefined },
      { filter: 'attributes.start_time > 1', pageToken: 'page-2' },
    ]);

    await act(async () => {
      await result.current.refetch();
    });
    expect(sent.slice(2)).toEqual([
      { filter: 'attributes.start_time > 2', pageToken: undefined },
      { filter: 'attributes.start_time > 2', pageToken: 'page-2' },
    ]);
  });

  test('sends the filter as given without resolveFilter', async () => {
    const sentFilters: string[] = [];
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', async (req, res, ctx) => {
        sentFilters.push((await req.json()).filter);
        return res(ctx.json({ runs: [] }));
      }),
    );
    renderHook(
      () => useExperimentEvaluationRunsData({ experimentId: 'test-experiment', enabled: true, filter: 'name = "x"' }),
      { wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider> },
    );
    await waitFor(() => expect(sentFilters).toEqual(['name = "x"']));
  });
});
