import type { QueryFunctionContext, UseQueryOptions } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { PromptVersionsForRunResponse, RegisteredPromptDetailsResponse, RegisteredPromptVersion } from '../types';
import { RegisteredPromptsApi } from '../api';

const queryFn = async ({ queryKey }: QueryFunctionContext<PromptVersionsForRunQueryKey>) => {
  const [, { runUuid }] = queryKey;
  return RegisteredPromptsApi.getPromptVersionsForRun(runUuid);
};

type PromptVersionsForRunQueryKey = ['run_uuid', { runUuid: string }];

export const usePromptVersionsForRunQuery = (
  { runUuid }: { runUuid: string },
  options: UseQueryOptions<
    PromptVersionsForRunResponse,
    Error,
    PromptVersionsForRunResponse,
    PromptVersionsForRunQueryKey
  > = {},
) => {
  const queryResult = useQuery<
    PromptVersionsForRunResponse,
    Error,
    PromptVersionsForRunResponse,
    PromptVersionsForRunQueryKey
  >(['run_uuid', { runUuid }], {
    queryFn,
    retry: false,
    ...options,
  });

  return {
    data: queryResult.data,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
