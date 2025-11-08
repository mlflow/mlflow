import { renderHook, waitFor } from '@testing-library/react';
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
});
