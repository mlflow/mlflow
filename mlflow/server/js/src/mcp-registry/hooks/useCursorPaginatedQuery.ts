import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import type { CursorPaginationProps } from '@databricks/design-system';
import { DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '../utils';

interface PaginatedResponse {
  next_page_token?: string;
}

export const useCursorPaginatedQuery = <TResponse extends PaginatedResponse, TData>({
  queryKeyPrefix,
  searchFilter,
  storageKey,
  queryFn,
  extractData,
  enabled,
}: {
  queryKeyPrefix: string;
  searchFilter?: string;
  storageKey: string;
  queryFn: (params: { searchFilter?: string; pageToken?: string; pageSize: number }) => Promise<TResponse>;
  extractData: (response: TResponse) => TData | undefined;
  enabled?: boolean;
}) => {
  const previousPageTokens = useRef<(string | undefined)[]>([]);
  const [currentPageToken, setCurrentPageToken] = useState<string | undefined>(undefined);

  const [pageSize, setPageSize] = useLocalStorage({
    key: storageKey,
    version: 0,
    initialValue: DEFAULT_PAGE_SIZE,
  });

  useEffect(() => {
    setCurrentPageToken(undefined);
    previousPageTokens.current = [];
  }, [searchFilter]);

  const pageSizeSelect = useMemo<CursorPaginationProps['pageSizeSelect']>(
    () => ({
      options: PAGE_SIZE_OPTIONS,
      default: pageSize,
      onChange(newPageSize) {
        setPageSize(newPageSize);
        setCurrentPageToken(undefined);
        previousPageTokens.current = [];
      },
    }),
    [pageSize, setPageSize],
  );

  const queryResult = useQuery<TResponse, Error>(
    [queryKeyPrefix, { searchFilter, pageToken: currentPageToken, pageSize }],
    {
      queryFn: () => queryFn({ searchFilter, pageToken: currentPageToken, pageSize }),
      retry: false,
      keepPreviousData: true,
      enabled,
    },
  );

  const onNextPage = useCallback(() => {
    if (queryResult.isFetching) return;
    previousPageTokens.current.push(currentPageToken);
    setCurrentPageToken(queryResult.data?.next_page_token ?? undefined);
  }, [queryResult.data?.next_page_token, queryResult.isFetching, currentPageToken]);

  const onPreviousPage = useCallback(() => {
    if (queryResult.isFetching) return;
    const previousPageToken = previousPageTokens.current.pop();
    setCurrentPageToken(previousPageToken);
  }, [queryResult.isFetching]);

  return {
    data: queryResult.data ? extractData(queryResult.data) : undefined,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    hasNextPage: Boolean(queryResult.data?.next_page_token),
    hasPreviousPage: Boolean(currentPageToken),
    onNextPage,
    onPreviousPage,
    pageSizeSelect,
    refetch: queryResult.refetch,
  };
};
