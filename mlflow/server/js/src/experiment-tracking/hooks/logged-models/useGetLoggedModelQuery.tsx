import { type QueryFunctionContext, useQueries, useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { LoggedModelProto } from '../../types';
import { loggedModelsDataRequest } from './request.utils';
import { useArrayMemo } from '../../../common/hooks/useArrayMemo';

type UseGetLoggedModelQueryResponseType = {
  model: LoggedModelProto;
};

type UseGetLoggedModelQueryKey = ['GET_LOGGED_MODEL', string];

const getQueryKey = (loggedModelId: string): UseGetLoggedModelQueryKey => ['GET_LOGGED_MODEL', loggedModelId] as const;

const queryFn = async ({
  queryKey: [, loggedModelId],
}: QueryFunctionContext<UseGetLoggedModelQueryKey>): Promise<UseGetLoggedModelQueryResponseType> =>
  loggedModelsDataRequest(`/ajax-api/2.0/mlflow/logged-models/${loggedModelId}`, 'GET');

/**
 * Retrieve logged model from API based on its ID
 */
export const useGetLoggedModelQuery = ({ loggedModelId }: { loggedModelId?: string }) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<
    UseGetLoggedModelQueryResponseType,
    Error,
    UseGetLoggedModelQueryResponseType,
    UseGetLoggedModelQueryKey
  >({
    queryKey: getQueryKey(loggedModelId ?? ''),
    queryFn,
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
  });

  return {
    isLoading,
    isFetching,
    data: data?.model,
    refetch,
    error,
  } as const;
};

/**
 * Retrieve multiple logged models from API based on their IDs
 */
export const useGetLoggedModelQueries = (loggedModelIds: string[] = []) => {
  const queries = useQueries({
    queries: loggedModelIds.map((modelId) => ({
      queryKey: getQueryKey(modelId),
      queryFn,
      cacheTime: 0,
      refetchOnWindowFocus: false,
      retry: false,
    })),
  });
  return useArrayMemo(queries);
};
