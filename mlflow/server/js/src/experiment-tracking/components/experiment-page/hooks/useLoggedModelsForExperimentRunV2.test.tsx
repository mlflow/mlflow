import { renderHook, waitFor } from '@testing-library/react';
import { useLoggedModelsForExperimentRunV2 } from './useLoggedModelsForExperimentRunV2';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

// Mock the useGetLoggedModelsQuery hook
jest.mock('../../../hooks/logged-models/useGetLoggedModelsQuery', () => ({
  useGetLoggedModelsQuery: jest.fn(),
}));

import { useGetLoggedModelsQuery } from '../../../hooks/logged-models/useGetLoggedModelsQuery';
import type { UseGetRunQueryResponseInputs, UseGetRunQueryResponseOutputs } from '../../run-page/hooks/useGetRunQuery';

describe('useLoggedModelsForExperimentRun', () => {
  // Create a wrapper component with QueryClientProvider
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
  );

  beforeEach(() => {
    jest.clearAllMocks();

    // Default mock implementation for useGetLoggedModelsQuery
    jest.mocked(useGetLoggedModelsQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: undefined,
    } as any);
  });

  test('should extract model IDs from inputs and outputs and fetch models', async () => {
    const mockModels = [
      { info: { model_id: 'model-123', name: 'model-model-123' }, data: {} },
      { info: { model_id: 'model-456', name: 'model-model-456' }, data: {} },
      { info: { model_id: 'model-789', name: 'model-model-789' }, data: {} },
    ];

    // Mock the query hook to return data
    jest.mocked(useGetLoggedModelsQuery).mockReturnValue({
      data: mockModels,
      isLoading: false,
      error: undefined,
    } as any);

    const runInputs = {
      modelInputs: [{ modelId: 'model-123' }, { modelId: 'model-456' }],
    } as UseGetRunQueryResponseInputs;

    const runOutputs = {
      modelOutputs: [
        { modelId: 'model-456' }, // Duplicate that should be de-duped
        { modelId: 'model-789' },
      ],
    } as UseGetRunQueryResponseOutputs;

    const { result } = renderHook(() => useLoggedModelsForExperimentRunV2({ runInputs, runOutputs }), { wrapper });

    // Verify the hook was called with the right parameters
    expect(useGetLoggedModelsQuery).toHaveBeenCalledWith(
      { modelIds: ['model-123', 'model-456', 'model-789'] },
      { enabled: true },
    );

    // Verify the returned data matches what we expect
    expect(result.current.models).toEqual(mockModels);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeUndefined();
  });

  test('should handle undefined inputs and outputs', async () => {
    const { result } = renderHook(
      () =>
        useLoggedModelsForExperimentRunV2({
          runInputs: undefined,
          runOutputs: undefined,
        }),
      { wrapper },
    );

    // Should not call useGetLoggedModelsQuery with modelIds
    expect(useGetLoggedModelsQuery).toHaveBeenCalledWith({ modelIds: undefined }, { enabled: false });

    // No models should be returned
    expect(result.current.models).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
  });

  test('should handle empty model inputs and outputs', async () => {
    const runInputs: UseGetRunQueryResponseInputs = {
      modelInputs: [],
    } as any;

    const runOutputs: UseGetRunQueryResponseOutputs = {
      modelOutputs: [],
    } as any;

    const { result } = renderHook(() => useLoggedModelsForExperimentRunV2({ runInputs, runOutputs }), { wrapper });

    // Should not call useGetLoggedModelsQuery with modelIds
    expect(useGetLoggedModelsQuery).toHaveBeenCalledWith({ modelIds: undefined }, { enabled: false });

    // No models should be returned
    expect(result.current.models).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
  });

  test('should handle API error', async () => {
    const mockError = new Error('API error');

    // Mock the query hook to return an error
    jest.mocked(useGetLoggedModelsQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: mockError,
    } as any);

    const runInputs: UseGetRunQueryResponseInputs = {
      modelInputs: [{ modelId: 'model-123' }],
    } as any;

    const { result } = renderHook(
      () =>
        useLoggedModelsForExperimentRunV2({
          runInputs,
          runOutputs: undefined,
        }),
      { wrapper },
    );

    // Verify error is propagated
    expect(result.current.error).toBe(mockError);
  });
});
