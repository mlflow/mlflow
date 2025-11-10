import {
  type QueryFunctionContext,
  useQuery,
  type UseQueryOptions,
} from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { LoggedModelProto } from '../../types';
import { chunk } from 'lodash';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

const LOGGED_MODEL_BY_ID_BATCH_LIMIT = 100; // API supports batch size of 100

type QueryResult = {
  models?: LoggedModelProto[];
};

type QueryKey = ['GET_LOGGED_MODELS', string[]];

const getQueryKey = (loggedModelIds: string[]): QueryKey => ['GET_LOGGED_MODELS', loggedModelIds] as const;

const queryFn = async ({ queryKey: [, loggedModelIds] }: QueryFunctionContext<QueryKey>): Promise<QueryResult[]> => {
  const modelIdChunks = chunk(loggedModelIds, LOGGED_MODEL_BY_ID_BATCH_LIMIT);
  return Promise.all<QueryResult>(
    modelIdChunks.map((chunkedIds) => {
      const queryParams = new URLSearchParams();
      for (const id of chunkedIds) {
        queryParams.append('model_ids', id);
      }
      return fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/logged-models:batchGet?${queryParams.toString()}`), 'GET');
    }),
  );
};

/**
 * Retrieve many logged model from API based on IDs
 */
export const useGetLoggedModelsQuery = (
  {
    modelIds,
  }: {
    modelIds?: string[];
  },
  options: UseQueryOptions<QueryResult[], Error, LoggedModelProto[], QueryKey>,
) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<QueryResult[], Error, LoggedModelProto[], QueryKey>({
    queryKey: getQueryKey(modelIds ?? []),
    queryFn,
    select: (results) => results?.flatMap((result) => result?.models || []),
    retry: false,
    ...options,
  });

  return {
    isLoading,
    isFetching,
    data,
    refetch,
    error,
  } as const;
};
