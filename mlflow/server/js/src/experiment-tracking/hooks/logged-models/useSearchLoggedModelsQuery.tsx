import { useInfiniteQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { last } from 'lodash';
import { LoggedModelProto } from '../../types';
import { loggedModelsDataRequest } from './request.utils';
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
  }: {
    experimentIds?: string[];
    orderByAsc?: boolean;
    orderByField?: string;
  },
  {
    enabled = true,
  }: {
    enabled?: boolean;
  } = {},
) => {
  // Uniquely identify the query by the experiment IDs, order by field, and order by asc
  const queryKey = ['SEARCH_LOGGED_MODELS', JSON.stringify(experimentIds), orderByField, orderByAsc];

  const { data, isLoading, isFetching, fetchNextPage, refetch, error } = useInfiniteQuery<
    UseSearchLoggedModelsQueryResponseType,
    Error
  >({
    queryKey,
    queryFn: async ({ pageParam }) => {
      const requestBody = {
        experiment_ids: experimentIds,
        order_by: [{ field_name: orderByField ?? 'creation_time', ascending: orderByAsc ?? false }],
        page_token: pageParam,
      };

      return loggedModelsDataRequest('/ajax-api/2.0/mlflow/logged-models/search', 'POST', requestBody);
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
