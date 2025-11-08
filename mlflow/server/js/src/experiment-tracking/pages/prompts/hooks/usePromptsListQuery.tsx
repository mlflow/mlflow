import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useCallback, useRef, useState } from 'react';
import type { RegisteredPromptsListResponse } from '../types';
import { RegisteredPromptsApi } from '../api';

const queryFn = ({ queryKey }: QueryFunctionContext<PromptsListQueryKey>) => {
  const [, { searchFilter, pageToken }] = queryKey;
  return RegisteredPromptsApi.listRegisteredPrompts(searchFilter, pageToken);
};

type PromptsListQueryKey = ['prompts_list', { searchFilter?: string; pageToken?: string }];

export const usePromptsListQuery = ({
  searchFilter,
}: {
  searchFilter?: string;
} = {}) => {
  const previousPageTokens = useRef<(string | undefined)[]>([]);

  const [currentPageToken, setCurrentPageToken] = useState<string | undefined>(undefined);

  const queryResult = useQuery<
    RegisteredPromptsListResponse,
    Error,
    RegisteredPromptsListResponse,
    PromptsListQueryKey
  >(['prompts_list', { searchFilter, pageToken: currentPageToken }], {
    queryFn,
    retry: false,
  });

  const onNextPage = useCallback(() => {
    previousPageTokens.current.push(currentPageToken);
    setCurrentPageToken(queryResult.data?.next_page_token);
  }, [queryResult.data?.next_page_token, currentPageToken]);

  const onPreviousPage = useCallback(() => {
    const previousPageToken = previousPageTokens.current.pop();
    setCurrentPageToken(previousPageToken);
  }, []);

  return {
    data: queryResult.data?.registered_models,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    hasNextPage: queryResult.data?.next_page_token !== undefined,
    hasPreviousPage: Boolean(currentPageToken),
    onNextPage,
    onPreviousPage,
    refetch: queryResult.refetch,
  };
};
