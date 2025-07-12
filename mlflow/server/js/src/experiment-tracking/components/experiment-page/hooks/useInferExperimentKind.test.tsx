import { renderHook, waitFor } from '@testing-library/react';
import { UseGetExperimentQueryResultExperiment } from '../../../hooks/useExperimentQuery';
import { useInferExperimentKind } from './useInferExperimentKind';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { ExperimentKind } from '../../../constants';

describe('useInferExperimentKind', () => {
  const server = setupServer();
  beforeEach(() => {
    server.resetHandlers();
    jest.clearAllMocks();
  });
  const renderTestHook = (props: Partial<Parameters<typeof useInferExperimentKind>[0]> = {}) => {
    const queryClient = new QueryClient();

    return renderHook(
      () =>
        useInferExperimentKind({
          experimentId: '123',
          isLoadingExperiment: false,
          updateExperimentKind: jest.fn(),
          ...props,
        }),
      {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      },
    );
  };

  test('it should not be able to infer any specific type when no traces or runs are present', async () => {
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        return res(ctx.json({}));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        return res(ctx.json({}));
      }),
    );
    const updateExperimentKind = jest.fn();
    const { result } = renderTestHook({ updateExperimentKind });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.inferredExperimentKind).toBe(ExperimentKind.NO_INFERRED_TYPE);
    expect(updateExperimentKind).not.toHaveBeenCalled();
  });

  test('it should infer GenAI type when traces are present', async () => {
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        return res(ctx.json({ traces: [{ id: 'trace1' }] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        return res(ctx.json({ runs: [{ info: { run_uuid: 'run1' } }] }));
      }),
    );
    const updateExperimentKind = jest.fn();
    const { result } = renderTestHook({ updateExperimentKind });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.inferredExperimentKind).toBe(ExperimentKind.GENAI_DEVELOPMENT_INFERRED);
    expect(updateExperimentKind).not.toHaveBeenCalled();
  });

  test('it should infer custom model development when no traces, but training runs are present', async () => {
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        return res(ctx.json({ traces: [] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        return res(ctx.json({ runs: [{ info: { run_uuid: 'run1' } }] }));
      }),
    );
    const updateExperimentKind = jest.fn();
    const { result } = renderTestHook({ updateExperimentKind });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.inferredExperimentKind).toBe(ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED);
    expect(updateExperimentKind).not.toHaveBeenCalled();
  });

  test('it should skip inference logic if the experiment is still loading', async () => {
    const tracesApiSpyFn = jest.fn();
    const searchRunsApiSpyFn = jest.fn();
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        tracesApiSpyFn(req);
        return res(ctx.json({ traces: [] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        searchRunsApiSpyFn(req);
        return res(ctx.json({ runs: [{ info: { run_uuid: 'run1' } }] }));
      }),
    );
    const updateExperimentKind = jest.fn();
    const { result } = renderTestHook({
      experimentId: undefined,
      isLoadingExperiment: true,
      updateExperimentKind,
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(true);
    });

    expect(tracesApiSpyFn).not.toHaveBeenCalled();
    expect(searchRunsApiSpyFn).not.toHaveBeenCalled();
    expect(updateExperimentKind).not.toHaveBeenCalled();
  });

  test('it should skip inference logic if the hook is disabled', async () => {
    const tracesApiSpyFn = jest.fn();
    const searchRunsApiSpyFn = jest.fn();
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        tracesApiSpyFn(req);
        return res(ctx.json({ traces: [] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        searchRunsApiSpyFn(req);
        return res(ctx.json({ runs: [{ info: { run_uuid: 'run1' } }] }));
      }),
    );
    const updateExperimentKind = jest.fn();
    const { result } = renderTestHook({
      enabled: false,
      updateExperimentKind,
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(tracesApiSpyFn).not.toHaveBeenCalled();
    expect(searchRunsApiSpyFn).not.toHaveBeenCalled();
    expect(updateExperimentKind).not.toHaveBeenCalled();
  });
});
