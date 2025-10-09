import { type QueryFunctionContext, useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { LoggedModelProto } from '../../types';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

type UseGetLoggedModelQueryResponseType = {
  model: LoggedModelProto;
};

type UseGetLoggedModelQueryKey = ['GET_LOGGED_MODEL', string];

const getQueryKey = (loggedModelId: string): UseGetLoggedModelQueryKey => ['GET_LOGGED_MODEL', loggedModelId] as const;

const queryFn = async ({
  queryKey: [, loggedModelId],
}: QueryFunctionContext<UseGetLoggedModelQueryKey>): Promise<UseGetLoggedModelQueryResponseType> =>
  fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/logged-models/${loggedModelId}`), 'GET');

/**
 * Retrieve logged model from API based on its ID
 */
export const useGetLoggedModelQuery = ({
  loggedModelId,
  enabled = true,
}: {
  loggedModelId?: string;
  enabled?: boolean;
}) => {
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
    enabled,
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
 * A non-hook version of useGetLoggedModelQuery that can be used in async functions.
 * @deprecated Use useGetLoggedModelQuery instead. This function is provided for backward compatibility for legacy class-based components.
 */
export const asyncGetLoggedModel = async (
  loggedModelId: string,
  failSilently = false,
): Promise<UseGetLoggedModelQueryResponseType | undefined> => {
  try {
    const data = await fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/logged-models/${loggedModelId}`), 'GET');
    return data;
  } catch (error) {
    if (failSilently) {
      return undefined;
    }
    throw error;
  }
};
