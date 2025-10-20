import { useInfiniteQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { isEmpty, last, uniqBy } from 'lodash';
import type { LoggedModelMetricDataset, LoggedModelProto } from '../../types';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMemo } from 'react';

type UseSearchLoggedModelsQueryResponseType = {
  models: LoggedModelProto[];
  next_page_token?: string;
};

export const useSearchLoggedModelsQuery = (
  {
    experimentIds,
    orderByAsc,
    orderByField,
    searchQuery,
    selectedFilterDatasets,
    orderByDatasetName,
    orderByDatasetDigest,
  }: {
    experimentIds?: string[];
    orderByAsc?: boolean;
    orderByField?: string;
    searchQuery?: string;
    selectedFilterDatasets?: LoggedModelMetricDataset[];
    orderByDatasetName?: string;
    orderByDatasetDigest?: string;
  },
  {
    enabled = true,
  }: {
    enabled?: boolean;
  } = {},
) => {
  // Uniquely identify the query by the experiment IDs, order by, filter query and datasets, and order by asc
  const queryKey = [
    'SEARCH_LOGGED_MODELS',
    JSON.stringify(experimentIds),
    orderByField,
    orderByAsc,
    searchQuery,
    JSON.stringify(selectedFilterDatasets),
    orderByDatasetName,
    orderByDatasetDigest,
  ];

  const { data, isLoading, isFetching, fetchNextPage, refetch, error } = useInfiniteQuery<
    UseSearchLoggedModelsQueryResponseType,
    Error
  >({
    queryKey,
    queryFn: async ({ pageParam }) => {
      const requestBody = {
        experiment_ids: experimentIds,
        order_by: [
          {
            field_name: orderByField ?? 'creation_time',
            ascending: orderByAsc ?? false,
            dataset_name: orderByDatasetName,
            dataset_digest: orderByDatasetDigest,
          },
        ],

        page_token: pageParam,
        filter: searchQuery,
        datasets: !isEmpty(selectedFilterDatasets) ? selectedFilterDatasets : undefined,
      };

      return fetchAPI(getAjaxUrl('ajax-api/2.0/mlflow/logged-models/search'), 'POST', requestBody);
    },
    cacheTime: 0,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  });

  // Concatenate all the models from all the result pages
  const modelsData = useMemo(() => data?.pages.flatMap((page) => page?.models).filter(Boolean), [data]);

  // The current page token is the one from the last page
  const nextPageToken = last(data?.pages)?.next_page_token;

  return {
    isLoading,
    isFetching,
    data: modelsData,
    nextPageToken,
    refetch,
    error,
    loadMoreResults: fetchNextPage,
  } as const;
};
