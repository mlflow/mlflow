import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useCallback, useMemo, useState } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import type { CursorPaginationProps } from '@databricks/design-system';

export const DEFAULT_PAGE_SIZE = 25;
export const PAGE_SIZE_OPTIONS = [10, 25, 50, 100];

interface PaginatedResponse {
  next_page_token?: string | null;
}

export const useCursorPaginatedQuery = <TResponse extends PaginatedResponse, TData>({
  queryKeyPrefix,
  searchFilter,
  extraQueryKeys,
  storageKey,
  queryFn,
  extractData,
  enabled,
  keepPreviousData = true,
}: {
  queryKeyPrefix: string;
  searchFilter?: string;
  extraQueryKeys?: Record<string, unknown>;
  storageKey: string;
  queryFn: (params: { searchFilter?: string; pageToken?: string; pageSize: number }) => Promise<TResponse>;
  extractData: (response: TResponse) => TData | undefined;
  enabled?: boolean;
  keepPreviousData?: boolean;
}) => {
  const paginationScope = JSON.stringify([searchFilter, extraQueryKeys]);
  const [paginationState, setPaginationState] = useState(() => ({
    scope: paginationScope,
    pageToken: undefined as string | undefined,
    previousPageTokens: [] as (string | undefined)[],
  }));

  if (paginationState.scope !== paginationScope) {
    setPaginationState({ scope: paginationScope, pageToken: undefined, previousPageTokens: [] });
  }
  const currentPageToken = paginationState.scope === paginationScope ? paginationState.pageToken : undefined;

  const [pageSize, setPageSize] = useLocalStorage({
    key: storageKey,
    version: 0,
    initialValue: DEFAULT_PAGE_SIZE,
  });

  const pageSizeSelect = useMemo<CursorPaginationProps['pageSizeSelect']>(
    () => ({
      options: PAGE_SIZE_OPTIONS,
      default: pageSize,
      onChange(newPageSize) {
        setPageSize(newPageSize);
        setPaginationState({ scope: paginationScope, pageToken: undefined, previousPageTokens: [] });
      },
    }),
    [pageSize, setPageSize, paginationScope],
  );

  const queryResult = useQuery<TResponse, Error>(
    [queryKeyPrefix, { searchFilter, pageToken: currentPageToken, pageSize, ...extraQueryKeys }],
    {
      queryFn: () => queryFn({ searchFilter, pageToken: currentPageToken, pageSize }),
      retry: false,
      keepPreviousData,
      enabled,
    },
  );

  const onNextPage = useCallback(() => {
    if (queryResult.isFetching) return;
    setPaginationState({
      scope: paginationScope,
      pageToken: queryResult.data?.next_page_token ?? undefined,
      previousPageTokens: [...paginationState.previousPageTokens, currentPageToken],
    });
  }, [
    queryResult.data?.next_page_token,
    queryResult.isFetching,
    currentPageToken,
    paginationScope,
    paginationState.previousPageTokens,
  ]);

  const onPreviousPage = useCallback(() => {
    if (queryResult.isFetching) return;
    const previousPageToken = paginationState.previousPageTokens[paginationState.previousPageTokens.length - 1];
    setPaginationState({
      scope: paginationScope,
      pageToken: previousPageToken,
      previousPageTokens: paginationState.previousPageTokens.slice(0, -1),
    });
  }, [queryResult.isFetching, paginationScope, paginationState.previousPageTokens]);

  return {
    data: queryResult.data ? extractData(queryResult.data) : undefined,
    rawResponse: queryResult.data,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    isFetching: queryResult.isFetching,
    hasNextPage: Boolean(queryResult.data?.next_page_token),
    hasPreviousPage: Boolean(currentPageToken),
    onNextPage,
    onPreviousPage,
    pageSizeSelect,
    refetch: queryResult.refetch,
  };
};
