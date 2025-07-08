import { useQuery, QueryFunctionContext, defaultContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { useCallback, useContext, useRef, useState } from 'react';
import { SearchExperimentsApiResponse } from '../../../types';
import { useLocalStorage } from '@mlflow/mlflow/src/shared/web-shared/hooks/useLocalStorage';
import { CursorPaginationProps } from '@databricks/design-system';
import { SortingState } from '@tanstack/react-table';

const STORE_KEY = {
  PAGE_SIZE: 'experiments_page.page_size',
  SORTING_STATE: 'experiments_page.sorting_state',
};
const DEFAULT_PAGE_SIZE = 25;

const ExperimentListQueryKeyHeader = 'experiment_list';

type ExperimentListQueryKey = [
  typeof ExperimentListQueryKeyHeader,
  { searchFilter?: string; pageToken?: string; pageSize?: number; sorting?: SortingState },
];

export const useInvalidateExperimentList = () => {
  const context = useContext(defaultContext);
  return () => {
    context?.invalidateQueries({ queryKey: [ExperimentListQueryKeyHeader] });
  };
};

const queryFn = ({ queryKey }: QueryFunctionContext<ExperimentListQueryKey>) => {
  const [, { searchFilter, pageToken, pageSize, sorting }] = queryKey;

  // NOTE: REST API docs are not detailed enough, see: mlflow/store/tracking/abstract_store.py#search_experiments
  const orderBy = sorting?.map((column) => ['order_by', `${column.id} ${column.desc ? 'DESC' : 'ASC'}`]) ?? [];

  const data = [['max_results', String(pageSize)], ...orderBy];

  if (searchFilter) {
    data.push(['filter', `name ILIKE '%${searchFilter}%'`]);
  }

  if (pageToken) {
    data.push(['page_token', pageToken]);
  }

  return MlflowService.searchExperiments(data);
};

export const useExperimentListQuery = ({ searchFilter }: { searchFilter?: string } = {}) => {
  const previousPageTokens = useRef<(string | undefined)[]>([]);

  const [currentPageToken, setCurrentPageToken] = useState<string | undefined>(undefined);

  const [pageSize, setPageSize] = useLocalStorage({
    key: STORE_KEY.PAGE_SIZE,
    version: 0,
    initialValue: DEFAULT_PAGE_SIZE,
  });

  const [sorting, setSorting] = useLocalStorage<SortingState>({
    key: STORE_KEY.SORTING_STATE,
    version: 0,
    initialValue: [{ id: 'last_update_time', desc: true }],
  });

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
  >([ExperimentListQueryKeyHeader, { searchFilter, pageToken: currentPageToken, pageSize, sorting }], {
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
    sorting,
    setSorting,
  };
};
