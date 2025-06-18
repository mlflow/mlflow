import { useQuery, QueryFunctionContext, defaultContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { useCallback, useContext, useRef, useState } from 'react';
import { SearchExperimentsApiResponse } from '../../../types';
import { useLocalStorage } from '@mlflow/mlflow/src/shared/web-shared/hooks/useLocalStorage';
import { CursorPaginationProps } from '@databricks/design-system';

const STORE_KEY = 'experiments_page_page_size';
const DEFAULT_PAGE_SIZE = 10;

const ExperimentListQueryKeyHeader = 'experiment_list';

type ExperimentListQueryKey = [
  typeof ExperimentListQueryKeyHeader,
  { searchFilter?: string; pageToken?: string; pageSize?: number },
];

export const useInvalidateExperimentList = () => {
  const context = useContext(defaultContext);
  return () => {
    context?.invalidateQueries({ queryKey: [ExperimentListQueryKeyHeader] });
  };
};

const queryFn = ({ queryKey }: QueryFunctionContext<ExperimentListQueryKey>) => {
  const [, { searchFilter, pageToken, pageSize }] = queryKey;
  return MlflowService.searchExperiments({
    filter: searchFilter,
    max_results: String(pageSize),
    page_token: pageToken,
  });
};

export const useExperimentListQuery = ({ searchFilter }: { searchFilter?: string } = {}) => {
  const previousPageTokens = useRef<(string | undefined)[]>([]);

  const [currentPageToken, setCurrentPageToken] = useState<string | undefined>(undefined);

  const [pageSize, setPageSize] = useLocalStorage({ key: STORE_KEY, version: 0, initialValue: DEFAULT_PAGE_SIZE });

  const pageSizeSelect: CursorPaginationProps['pageSizeSelect'] = {
    options: [10, 25, 50, 100],
    default: pageSize,
    onChange(pageSize) {
      setPageSize(pageSize);
      setCurrentPageToken(undefined);
    },
  };

  const queryResult = useQuery<
    SearchExperimentsApiResponse,
    Error,
    SearchExperimentsApiResponse,
    ExperimentListQueryKey
  >([ExperimentListQueryKeyHeader, { searchFilter, pageToken: currentPageToken, pageSize }], {
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
    data: queryResult.data?.experiments,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    hasNextPage: queryResult.data?.next_page_token !== undefined,
    hasPreviousPage: Boolean(currentPageToken),
    onNextPage,
    onPreviousPage,
    refetch: queryResult.refetch,
    pageSizeSelect,
  };
};
