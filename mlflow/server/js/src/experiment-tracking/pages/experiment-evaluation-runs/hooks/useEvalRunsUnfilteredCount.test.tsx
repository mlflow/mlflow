import { describe, beforeAll, beforeEach, test, expect } from '@jest/globals';

import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../../constants';
import { useEvalRunsUnfilteredCount } from './useEvalRunsUnfilteredCount';

const detectionRun = (runUuid: string) => ({
  info: { run_uuid: runUuid, experiment_id: 'exp-1' },
  data: { tags: [{ key: MLFLOW_RUN_TYPE_TAG, value: MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION }] },
});
const evalRun = (runUuid: string) => ({
  info: { run_uuid: runUuid, experiment_id: 'exp-1' },
  data: { tags: [] },
});

describe('useEvalRunsUnfilteredCount', () => {
  const server = setupServer();

  beforeAll(() => server.listen());

  beforeEach(() => server.resetHandlers());

  const renderCount = () =>
    renderHook(() => useEvalRunsUnfilteredCount({ experimentId: 'exp-1', enabled: true }), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });

  test('counts evaluation runs and ignores detection and training runs', async () => {
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) =>
        res(
          ctx.json({
            runs: [
              detectionRun('detection-1'),
              evalRun('eval-1'),
              {
                info: { run_uuid: 'training-1' },
                outputs: { model_outputs: [{ model_id: 'm-1' }] },
                data: { tags: [] },
              },
            ],
          }),
        ),
      ),
    );

    const { result } = renderCount();

    await waitFor(() => expect(result.current.runs).toHaveLength(1));
    expect(result.current.runs[0].info.runUuid).toBe('eval-1');
  });

  test('pages past a first page of only detection runs instead of counting zero', async () => {
    // Counting zero here would read as "this experiment has no runs at all", which answers an
    // active filter with first-run onboarding instead of the filtered-empty state and its reset.
    const firstPage = { runs: [detectionRun('detection-1'), detectionRun('detection-2')], next_page_token: 'page-2' };
    const secondPage = { runs: [evalRun('eval-1')] };
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', async (req, res, ctx) => {
        const body = (await req.json()) as { page_token?: string };
        return res(ctx.json(body.page_token === 'page-2' ? secondPage : firstPage));
      }),
    );

    const { result } = renderCount();

    await waitFor(() => expect(result.current.runs).toHaveLength(1));
    expect(result.current.runs[0].info.runUuid).toBe('eval-1');
  });

  test('reports an experiment with no runs as empty', async () => {
    server.use(rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => res(ctx.json({ runs: [] }))));

    const { result } = renderCount();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.runs).toEqual([]);
    expect(result.current.hasNextPage).toBe(false);
  });
});
