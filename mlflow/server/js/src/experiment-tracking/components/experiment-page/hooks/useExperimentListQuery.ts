import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { useCallback, useRef, useState } from 'react';
import type { SearchExperimentsApiResponse } from '../../../types';
import { useLocalStorage } from '@mlflow/mlflow/src/shared/web-shared/hooks/useLocalStorage';
import type { CursorPaginationProps } from '@databricks/design-system';
import type { SortingState } from '@tanstack/react-table';
import type { TagFilter } from './useTagsFilter';

const STORE_KEY = {
  PAGE_SIZE: 'experiments_page.page_size',
  SORTING_STATE: 'experiments_page.sorting_state',
};
const DEFAULT_PAGE_SIZE = 25;

const ExperimentListQueryKeyHeader = 'experiment_list';

type ExperimentListQueryKey = [
  typeof ExperimentListQueryKeyHeader,
  { searchFilter?: string; tagsFilter?: TagFilter[]; pageToken?: string; pageSize?: number; sorting?: SortingState },
];

export const useInvalidateExperimentList = () => {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: [ExperimentListQueryKeyHeader] });
  };
};

function tagFilterToSql(tagFilter: TagFilter) {
  switch (tagFilter.operator) {
    case 'IS':
      return `tags.\`${tagFilter.key}\` = '${tagFilter.value}'`;
    case 'IS NOT':
      return `tags.\`${tagFilter.key}\` != '${tagFilter.value}'`;
    case 'CONTAINS':
      return `tags.\`${tagFilter.key}\` ILIKE '%${tagFilter.value}%'`;
  }
}

function getFilters({ searchFilter, tagsFilter }: Pick<ExperimentListQueryKey['1'], 'searchFilter' | 'tagsFilter'>) {
  const filters = [];

  if (searchFilter) {
    filters.push(`name ILIKE '%${searchFilter}%'`);
  }

  for (const tagFilter of tagsFilter ?? []) {
    filters.push(tagFilterToSql(tagFilter));
  }

  if (filters.length > 0) {
    return ['filter', filters.join(' AND ')];
  } else {
    return undefined;
  }
}

const queryFn = ({ queryKey }: QueryFunctionContext<ExperimentListQueryKey>) => {
  const [, { searchFilter, tagsFilter, pageToken, pageSize, sorting }] = queryKey;

  // NOTE: REST API docs are not detailed enough, see: mlflow/store/tracking/abstract_store.py#search_experiments
  const orderBy = sorting?.map((column) => ['order_by', `${column.id} ${column.desc ? 'DESC' : 'ASC'}`]) ?? [];

  const data: (string[] | undefined)[] = [['max_results', String(pageSize)], ...orderBy];

  // NOTE: undefined values are fine, they're filtered out by `getBigIntJson` inside `MlflowService`
  data.push(getFilters({ searchFilter, tagsFilter }));

  if (pageToken) {
    data.push(['page_token', pageToken]);
  }

  return MlflowService.searchExperiments(data);
};

export const useExperimentListQuery = ({
  searchFilter,
  tagsFilter,
}: { searchFilter?: string; tagsFilter?: TagFilter[] } = {}) => {
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
  >([ExperimentListQueryKeyHeader, { searchFilter, tagsFilter, pageToken: currentPageToken, pageSize, sorting }], {
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
