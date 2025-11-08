import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { renderHook, waitFor } from '@testing-library/react';
import { useGetLoggedModelsQuery } from './useGetLoggedModelsQuery';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

describe('useGetLoggedModelsQuery', () => {
  const callSpy = jest.fn();

  setupServer(
    rest.get('/ajax-api/2.0/mlflow/logged-models:batchGet', (req, res, ctx) => {
      // Capture the model_ids from the URL query parameters
      const modelIds = req.url.searchParams.getAll('model_ids');
      callSpy(modelIds);

      return res(
        ctx.json({
          models: modelIds.map((id) => ({
            info: { model_id: id, name: `model-${id}` },
            data: {},
          })),
        }),
      );
    }),
  );

  beforeEach(() => {
    callSpy.mockClear();
  });

  test('should properly construct URL query params with model_ids', async () => {
    const modelIds = ['model-123', 'model-456', 'model-789'];

    const { result } = renderHook(() => useGetLoggedModelsQuery({ modelIds }, {}), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });

    // Wait for the query to complete
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Verify the correct model_ids were passed in the URL
    expect(callSpy).toHaveBeenCalledTimes(1);
    expect(callSpy).toHaveBeenCalledWith(modelIds);

    // Verify the returned data matches what we expect
    expect(result.current.data).toHaveLength(3);
    expect(result.current.data?.map((model) => model.info?.model_id)).toEqual(modelIds);
    expect(result.current.data?.map((model) => model.info?.name)).toEqual([
      'model-model-123',
      'model-model-456',
      'model-model-789',
    ]);
  });

  test('should handle empty model IDs array', async () => {
    const { result } = renderHook(() => useGetLoggedModelsQuery({ modelIds: [] }, {}), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(callSpy).toHaveBeenCalledTimes(0);
    expect(result.current.data).toEqual([]);
  });

  test('should handle undefined model IDs', async () => {
    const { result } = renderHook(() => useGetLoggedModelsQuery({}, {}), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(callSpy).toHaveBeenCalledTimes(0);
    expect(result.current.data).toEqual([]);
  });

  test('should chunk requests when requesting lot of model IDs', async () => {
    const threeAndHalfHundredModelIds = Array.from({ length: 350 }, (_, i) => `model-${i + 1}`);
    const { result } = renderHook(() => useGetLoggedModelsQuery({ modelIds: threeAndHalfHundredModelIds }, {}), {
      wrapper: ({ children }) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>,
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(callSpy).toHaveBeenCalledTimes(4);
    expect(callSpy).toHaveBeenCalledWith(threeAndHalfHundredModelIds.slice(0, 100));
    expect(callSpy).toHaveBeenCalledWith(threeAndHalfHundredModelIds.slice(100, 200));
    expect(callSpy).toHaveBeenCalledWith(threeAndHalfHundredModelIds.slice(200, 300));
    expect(callSpy).toHaveBeenCalledWith(threeAndHalfHundredModelIds.slice(300, 400));

    expect(result.current.data).toEqual(
      threeAndHalfHundredModelIds.map((id) => ({
        info: { model_id: id, name: `model-${id}` },
        data: {},
      })),
    );
  });
});
