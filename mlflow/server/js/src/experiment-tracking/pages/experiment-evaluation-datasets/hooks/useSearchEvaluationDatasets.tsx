import { useInfiniteQuery } from '@tanstack/react-query';
import { EvaluationDataset } from '../types';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMemo } from 'react';
import { SEARCH_EVALUATION_DATASETS_QUERY_KEY } from '../constants';

const SEARCH_EVALUATION_DATASETS_PAGE_SIZE = 50;

type SearchEvaluationDatasetsResponse = {
  datasets?: EvaluationDataset[];
  next_page_token?: string;
};

export const useSearchEvaluationDatasets = ({
  experimentId,
  enabled = true,
  nameFilter = '',
}: {
  experimentId: string;
  enabled?: boolean;
  nameFilter?: string;
}) => {
  const { data, fetchNextPage, hasNextPage, isLoading, isFetching, refetch, error } = useInfiniteQuery<
    SearchEvaluationDatasetsResponse,
    Error
  >({
    queryKey: [SEARCH_EVALUATION_DATASETS_QUERY_KEY, experimentId, nameFilter],
    queryFn: async ({ queryKey: [, experimentId, nameFilter], pageParam = undefined }) => {
      const filterString = nameFilter ? `name ILIKE '%${nameFilter}%'` : undefined;
      const requestBody = {
        experiment_ids: [experimentId],
        filter_string: filterString,
        order_by: ['created_time DESC'],
        max_results: SEARCH_EVALUATION_DATASETS_PAGE_SIZE,
        page_token: pageParam,
      };

      return (await fetchAPI(
        getAjaxUrl('ajax-api/3.0/mlflow/datasets/search'),
        'POST',
        requestBody,
      )) as SearchEvaluationDatasetsResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
  });

  const flatData = useMemo(() => data?.pages.flatMap((page) => page.datasets ?? []) ?? [], [data]);

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
