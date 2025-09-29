import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { useLoggedModelsForExperimentRunsTableV2 } from './useLoggedModelsForExperimentRunsTableV2';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { RunInfoEntity } from '../../../types';

// Enable feature flags
jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  isExperimentLoggedModelsUIEnabled: jest.fn(() => true),
}));

describe('useLoggedModelsForExperimentRunsTableV2', () => {
  const server = setupServer();

  beforeAll(() => server.listen());

  beforeEach(() => {
    server.use(
      rest.get('/ajax-api/2.0/mlflow/logged-models:batchGet', (req, res, ctx) => {
        // Extract model IDs from the query parameters
        const modelIds = req.url.searchParams.getAll('model_ids');

        const sourceRunIdByModelIdMap: Record<string, string> = {
          'model-id-1': 'run-1',
          'model-id-2': 'run-1',
          'model-id-3': 'run-2',
        };

        // Create mock models based on the requested IDs
        const models = modelIds.map((modelId) => ({
          info: {
            model_id: modelId,
            name: `model-${modelId}`,
            source_run_id: sourceRunIdByModelIdMap[modelId],
            experiment_id: 'test-experiment',
          },
          data: {},
        }));

        return res(ctx.json({ models }));
      }),
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  afterAll(() => {
    server.close();
  });

  // Create a wrapper component with QueryClientProvider
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
  );

  test('should extract model IDs from inputs and outputs and fetch models', async () => {
    // Prepare test data
    const runData = [
      {
        runInfo: { runUuid: 'run-1' } as RunInfoEntity,
        inputs: {
          modelInputs: [{ modelId: 'model-id-1' }],
        },
        outputs: {
          modelOutputs: [{ modelId: 'model-id-2' }],
        },
      },
      {
        runInfo: { runUuid: 'run-2' } as RunInfoEntity,
        inputs: {
          modelInputs: [{ modelId: 'model-id-3' }],
        },
        outputs: undefined,
      },
    ];

    // Render the hook
    const { result } = renderHook(() => useLoggedModelsForExperimentRunsTableV2({ runData, enabled: true }), {
      wrapper,
    });

    // Initially the result should be an empty object
    expect(result.current).toEqual({});

    // Wait for the query to complete
    await waitFor(() => {
      // Verify the hook returns the expected data structure
      expect(Object.keys(result.current)).toContain('run-1');
      expect(Object.keys(result.current)).toContain('run-2');
    });

    // Verify the models are correctly associated with their runs
    expect(result.current['run-1'].length).toBe(2);
    expect(result.current['run-1'][0].info?.model_id).toBe('model-id-1');
    expect(result.current['run-1'][1].info?.model_id).toBe('model-id-2');

    expect(result.current['run-2'].length).toBe(1);
    expect(result.current['run-2'][0].info?.model_id).toBe('model-id-3');
  });

  test('should handle runs with no model IDs', async () => {
    // Prepare test data with no model IDs
    const runData = [
      {
        runInfo: { runUuid: 'run-1' } as RunInfoEntity,
        inputs: { modelInputs: [] },
        outputs: { modelOutputs: [] },
      },
    ];

    // Render the hook
    const { result } = renderHook(() => useLoggedModelsForExperimentRunsTableV2({ runData, enabled: true }), {
      wrapper,
    });

    // The result should be an empty object since there are no models to fetch
    expect(result.current).toEqual({});

    // Wait for a short time to ensure no API calls are made
    await new Promise((resolve) => setTimeout(resolve, 100));

    // The result should still be an empty object
    expect(result.current).toEqual({});
  });

  test('should handle undefined inputs and outputs', async () => {
    // Prepare test data with undefined inputs and outputs
    const runData = [
      {
        runInfo: { runUuid: 'run-1' } as RunInfoEntity,
        // No inputs or outputs defined
      },
    ];

    // Render the hook
    const { result } = renderHook(() => useLoggedModelsForExperimentRunsTableV2({ runData, enabled: true }), {
      wrapper,
    });

    // The result should be an empty object since there are no models to fetch
    expect(result.current).toEqual({});

    // Wait for a short time to ensure no API calls are made
    await new Promise((resolve) => setTimeout(resolve, 100));

    // The result should still be an empty object
    expect(result.current).toEqual({});
  });

  test('should not fetch models when disabled', async () => {
    // Prepare test data
    const runData = [
      {
        runInfo: { runUuid: 'run-1' } as RunInfoEntity,
        inputs: {
          modelInputs: [{ modelId: 'model-id-1' }],
        },
      },
    ];

    // Render the hook with enabled=false
    const { result } = renderHook(() => useLoggedModelsForExperimentRunsTableV2({ runData, enabled: false }), {
      wrapper,
    });

    // The result should be an empty object
    expect(result.current).toEqual({});

    // Wait for a short time to ensure no API calls are made
    await new Promise((resolve) => setTimeout(resolve, 100));

    // The result should still be an empty object
    expect(result.current).toEqual({});
  });

  test('should deduplicate model IDs from the same run', async () => {
    // Prepare test data with duplicate model IDs
    const runData = [
      {
        runInfo: { runUuid: 'run-1' } as RunInfoEntity,
        inputs: {
          modelInputs: [{ modelId: 'model-id-1' }, { modelId: 'model-id-1' }], // Duplicate ID
        },
        outputs: {
          modelOutputs: [{ modelId: 'model-id-1' }, { modelId: 'model-id-2' }], // Another duplicate
        },
      },
    ];

    // Render the hook
    const { result } = renderHook(() => useLoggedModelsForExperimentRunsTableV2({ runData, enabled: true }), {
      wrapper,
    });

    // Wait for the query to complete
    await waitFor(() => {
      expect(Object.keys(result.current)).toContain('run-1');
    });

    // Verify the models are correctly deduplicated
    expect(result.current['run-1'].length).toBe(2); // Only 2 unique models

    // Check that the model IDs are unique
    const modelIds = result.current['run-1'].map((model) => model.info?.model_id);
    expect(new Set(modelIds).size).toBe(modelIds.length);
  });
});
