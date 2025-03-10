import { QueryFunction, useQuery, UseQueryOptions } from '@tanstack/react-query';
import type { RegisteredPromptDetailsResponse } from '../types';
import { RegisteredPromptsApi } from '../api';

const queryFn: QueryFunction<RegisteredPromptDetailsResponse, PromptDetailsQueryKey> = async ({ queryKey }) => {
  const [, { promptName }] = queryKey;
  const [detailsResponse, versionsResponse] = await Promise.all([
    RegisteredPromptsApi.getPromptDetails(promptName),
    RegisteredPromptsApi.getPromptVersions(promptName),
  ]);

  return {
    prompt: detailsResponse.registered_model,
    versions: versionsResponse.model_versions ?? [],
  };
};

type PromptDetailsQueryKey = ['prompt_details', { promptName: string }];

export const usePromptDetailsQuery = (
  { promptName }: { promptName: string },
  options: UseQueryOptions<
    RegisteredPromptDetailsResponse,
    Error,
    RegisteredPromptDetailsResponse,
    PromptDetailsQueryKey
  > = {},
) => {
  const queryResult = useQuery<
    RegisteredPromptDetailsResponse,
    Error,
    RegisteredPromptDetailsResponse,
    PromptDetailsQueryKey
  >(['prompt_details', { promptName }], {
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
