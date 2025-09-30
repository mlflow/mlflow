import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { EvaluationDataset } from '../types';
import { postJson } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMemo } from 'react';

const SEARCH_EVALUATION_DATASETS_PAGE_SIZE = 20;

type SearchEvaluationDatasetsResponse = {
  datasets: EvaluationDataset[];
  next_page_token: string;
};

export const useSearchEvaluationDatasets = ({
  experimentId,
  enabled = true,
  filter = '',
}: {
  experimentId: string;
  enabled?: boolean;
  filter?: string;
}) => {
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, refetch, error } = useInfiniteQuery<
    SearchEvaluationDatasetsResponse,
    Error
  >({
    queryKey: ['SEARCH_EVALUATION_DATASETS', experimentId, filter],
    queryFn: async ({ queryKey: [, experimentId, filter], pageParam = undefined }) => {
      const requestBody = {
        experiment_ids: [experimentId],
        filter,
        order_by: ['created_time DESC'],
        max_results: SEARCH_EVALUATION_DATASETS_PAGE_SIZE,
        page_token: pageParam,
      };

      return (await postJson({
        relativeUrl: 'ajax-api/3.0/mlflow/datasets/search',
        data: requestBody,
      })) as SearchEvaluationDatasetsResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
  });

  const flatData = useMemo(() => data?.pages.flatMap((page) => page.datasets), [data]);

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
