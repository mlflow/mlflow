import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { renderHook, waitFor } from '@testing-library/react';
import { useRelatedRunsDataForLoggedModels } from './useRelatedRunsDataForLoggedModels';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

describe('useRelatedRunsDataForLoggedModels', () => {
  const callSpy = jest.fn();
  setupServer(
    rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
      callSpy(req.url.searchParams.get('run_id'));
      return res(
        ctx.json({
          run: {
            info: {
              run_name: `run-name-for-${req.url.searchParams.get('run_id')}`,
              run_id: req.url.searchParams.get('run_id'),
            },
          },
        }),
      );
    }),
  );
  beforeEach(() => {
    callSpy.mockClear();
  });
  test('should properly call for runs data based on metadata found in logged models', async () => {
    const { result } = renderHook(
      () =>
        useRelatedRunsDataForLoggedModels({
          loggedModels: [
            {
              data: {
                metrics: [{ run_id: 'logged-metric-run-id-3' }, { run_id: 'logged-metric-run-id-2' }],
              },
              info: {
                source_run_id: 'source-run-id-1',
                experiment_id: 'experimentId',
              },
            } as any,
            {
              data: {
                metrics: [{ run_id: 'logged-metric-run-id-1' }],
              },
              info: {
                source_run_id: 'source-run-id-100',
                experiment_id: 'experimentId',
              },
            } as any,
          ],
        }),
      {
        wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
      },
    );

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(callSpy).toHaveBeenCalledTimes(5);

    expect(callSpy).toHaveBeenCalledWith('logged-metric-run-id-1');
    expect(callSpy).toHaveBeenCalledWith('logged-metric-run-id-2');
    expect(callSpy).toHaveBeenCalledWith('logged-metric-run-id-3');
    expect(callSpy).toHaveBeenCalledWith('source-run-id-1');
    expect(callSpy).toHaveBeenCalledWith('source-run-id-100');

    expect(result.current.data.map(({ info: { runName } }) => runName)).toEqual([
      'run-name-for-logged-metric-run-id-1',
      'run-name-for-logged-metric-run-id-2',
      'run-name-for-logged-metric-run-id-3',
      'run-name-for-source-run-id-1',
      'run-name-for-source-run-id-100',
    ]);
  });

  test('should not call for runs data based when not necessary', async () => {
    const { result } = renderHook(
      () =>
        useRelatedRunsDataForLoggedModels({
          loggedModels: [
            {
              data: {},
              info: {
                source_run_id: '',
                experiment_id: 'experimentId',
              },
            } as any,
          ],
        }),
      {
        wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
      },
    );

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(callSpy).not.toHaveBeenCalled();
  });
});
