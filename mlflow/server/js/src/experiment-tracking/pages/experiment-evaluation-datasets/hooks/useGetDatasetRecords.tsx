import { useInfiniteQuery } from '@tanstack/react-query';
import { fetchAPI, getAjaxUrl, getJson } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMemo } from 'react';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { EvaluationDatasetRecord } from '../types';
import { GET_DATASET_RECORDS_QUERY_KEY } from '../constants';

const GET_DATASET_RECORDS_PAGE_SIZE = 50;

type GetDatasetRecordsResponse = {
  // JSON serialized list of dataset records
  records: string;
  next_page_token?: string;
};

export const useGetDatasetRecords = ({ datasetId, enabled = true }: { datasetId: string; enabled?: boolean }) => {
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, refetch, error } = useInfiniteQuery<
    GetDatasetRecordsResponse,
    Error
  >({
    queryKey: [GET_DATASET_RECORDS_QUERY_KEY, datasetId],
    queryFn: async ({ queryKey: [, datasetId], pageParam = undefined }) => {
      const queryParams = new URLSearchParams();
      queryParams.set('dataset_id', datasetId as string);
      queryParams.set('max_results', GET_DATASET_RECORDS_PAGE_SIZE.toString());
      if (pageParam) {
        queryParams.set('page_token', pageParam ?? '');
      }

      return (await fetchAPI(
        getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records?${queryParams.toString()}`),
        'GET',
      )) as GetDatasetRecordsResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
  });

  const flatData = useMemo(
    () => data?.pages.flatMap((page) => parseJSONSafe(page.records) as EvaluationDatasetRecord[]) ?? [],
    [data],
  );

  return {
    data: flatData,
    fetchNextPage,
    hasNextPage,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
