import { renderHook, waitFor } from '@testing-library/react';
import { useLoggedModelsForExperimentRun } from './useLoggedModelsForExperimentRun';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { QueryClientProvider, QueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import React from 'react';

const mockExperimentId = 'experiment-123';
const mockRunId = 'run-456';
const mockModelId1 = 'model-789';
const mockModelId2 = 'model-101';

const mockRunInputs = {
  datasetInputs: [],
  modelInputs: [{ modelId: mockModelId1 }],
} as any;

const mockRunOutputs = {
  modelOutputs: [{ modelId: mockModelId2 }],
} as any;

const mockLoggedModelsData = [
  { id: 'logged-model-1', name: 'Model 1', attributes: { model_id: mockModelId1 } },
  { id: 'logged-model-2', name: 'Model 2', attributes: { model_id: mockModelId2 } },
];

// Setup MSW server to mock API calls
const server = setupServer(
  rest.post('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) => {
    return res(
      ctx.json({
        models: mockLoggedModelsData,
        next_page_token: null,
      }),
    );
  }),
);

// Create a wrapper with QueryClient for the hooks
const createHookWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useLoggedModelsForExperimentRun', () => {
  beforeAll(() => server.listen());
  afterEach(() => {
    server.resetHandlers();
    jest.clearAllMocks();
  });
  afterAll(() => server.close());

  test('should disable the hook when inputs change from valid to empty', async () => {
    // Create a hook with valid inputs/outputs initially
    const { result, rerender } = renderHook(
      (props) => useLoggedModelsForExperimentRun(mockExperimentId, mockRunId, props.inputs, props.outputs),
      {
        wrapper: createHookWrapper(),
        initialProps: {
          inputs: mockRunInputs,
          outputs: mockRunOutputs,
        },
      },
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.models).toEqual(mockLoggedModelsData);
    });

    rerender({
      inputs: { modelInputs: [] },
      outputs: { modelOutputs: [] },
    });

    // The hook should now be disabled and return no data
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.models).toBeUndefined();
  });
});
