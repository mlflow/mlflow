import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useCreateEvaluationDatasetMutation } from './useCreateEvaluationDatasetMutation';
import { V2_DATASETS_PAGE_QUERY_KEY } from '../../experiment-evaluation-datasets-v2/hooks/useDatasetsPageQuery';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

describe('useCreateEvaluationDatasetMutation', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  test('invalidates the v2 datasets list so the new dataset shows up', async () => {
    jest.mocked(fetchAPI).mockResolvedValue({ dataset: { dataset_id: 'd-1', name: 'my_ds' } });
    const pageKey = [V2_DATASETS_PAGE_QUERY_KEY, 'exp-1', '', 25, undefined];
    queryClient.setQueryData(pageKey, { datasets: [] });
    const onSuccess = jest.fn();

    const { result } = renderHook(() => useCreateEvaluationDatasetMutation({ onSuccess }), { wrapper });
    act(() => {
      result.current.createEvaluationDatasetMutation({ datasetName: 'my_ds', experimentIds: ['exp-1'] });
    });

    await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    expect(queryClient.getQueryState(pageKey)?.isInvalidated).toBe(true);
  });
});
