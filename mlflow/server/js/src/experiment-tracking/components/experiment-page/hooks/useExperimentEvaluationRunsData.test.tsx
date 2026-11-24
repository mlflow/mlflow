import { describe, beforeAll, beforeEach, test, expect } from '@jest/globals';

import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { useExperimentEvaluationRunsData } from './useExperimentEvaluationRunsData';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../../constants';
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

  const detectionRun = (runUuid: string) => ({
    info: { run_uuid: runUuid, name: runUuid, experiment_id: 'test-experiment' },
    data: { tags: [{ key: MLFLOW_RUN_TYPE_TAG, value: MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION }] },
  });
  const evalRun = (runUuid: string) => ({
    info: { run_uuid: runUuid, name: runUuid, experiment_id: 'test-experiment' },
    data: { tags: [] },
  });

  const renderRunsHook = (excludeIssueDetectionRuns: boolean) =>
    renderHook(
      () =>
        useExperimentEvaluationRunsData({
          experimentId: 'test-experiment',
          enabled: true,
          filter: '',
          excludeIssueDetectionRuns,
        }),
      {
        wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
      },
    );

  test('leaves issue detection runs out of the list when asked to', async () => {
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) =>
        res(ctx.json({ runs: [detectionRun('detection-1'), evalRun('eval-1')] })),
      ),
    );

    const { result } = renderRunsHook(true);

    await waitFor(() => expect(result.current.data).toHaveLength(1));
    expect(result.current.data?.[0]).toEqual(
      expect.objectContaining({ info: expect.objectContaining({ run_uuid: 'eval-1' }) }),
    );
  });

  test('keeps paging when a whole page is issue detection runs, so the list is not empty', async () => {
    // Without this the first page filters down to nothing and the list renders empty even though
    // the experiment has evaluation runs on the next page.
    const firstPage = {
      runs: [detectionRun('detection-1'), detectionRun('detection-2')],
      next_page_token: 'page-2',
    };
    const secondPage = { runs: [evalRun('eval-1')] };
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', async (req, res, ctx) => {
        const body = (await req.json()) as { page_token?: string };
        return res(ctx.json(body.page_token === 'page-2' ? secondPage : firstPage));
      }),
    );

    const { result } = renderRunsHook(true);

    await waitFor(() => expect(result.current.data).toHaveLength(1));
    expect(result.current.data?.[0]).toEqual(
      expect.objectContaining({ info: expect.objectContaining({ run_uuid: 'eval-1' }) }),
    );
  });

  test('lists issue detection runs when not asked to exclude them', async () => {
    server.use(
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) =>
        res(ctx.json({ runs: [detectionRun('detection-1'), evalRun('eval-1')] })),
      ),
    );

    const { result } = renderRunsHook(false);

    await waitFor(() => expect(result.current.data).toHaveLength(2));
  });
});
