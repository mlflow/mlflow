import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { Dataset } from './useDatasetsQueries';
import { DEFAULT_DATASETS_ORDER_BY } from '../utils/constants';

export const V2_DATASETS_PAGE_QUERY_KEY = 'v2ListDatasetsPage';

interface ListDatasetsPageResponse {
  datasets?: Dataset[];
  next_page_token?: string;
}

interface UseDatasetsPageQueryParams {
  experimentId: string;
  /** Server-side substring filter on the dataset name. Empty string disables the filter. */
  nameFilter: string;
  pageSize: number;
  /** Opaque cursor from a previous response, or `undefined` for the first page. */
  pageToken: string | undefined;
}

const fetchDatasetsPage = async ({
  experimentId,
  nameFilter,
  pageSize,
  pageToken,
}: UseDatasetsPageQueryParams): Promise<ListDatasetsPageResponse> => {
  // OSS speaks the v3 search endpoint with a JSON body — different from universe's
  // GET-with-filter-string managed-evals endpoint. Filter syntax is also different
  // (`name ILIKE`), so `buildFilterString` is unused here; we pass the raw nameFilter
  // through and let the OSS handler build its own filter.
  const trimmedFilter = nameFilter.trim();
  const filterString = trimmedFilter ? `name ILIKE '%${trimmedFilter.replace(/'/g, "''")}%'` : undefined;
  return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/datasets/search'), {
    method: 'POST',
    body: {
      experiment_ids: [experimentId],
      filter_string: filterString,
      order_by: [DEFAULT_DATASETS_ORDER_BY],
      max_results: pageSize,
      page_token: pageToken,
    },
  })) as ListDatasetsPageResponse;
};

/**
 * Paginated, server-filtered datasets query. Unlike the legacy `useListDatasetsQuery`
 * (which loops to fetch every dataset), this issues a single request per page so the UI
 * stays responsive on workspaces with many datasets.
 *
 * Cache: `staleTime: Infinity` so navigating away and back doesn't refetch — the
 * list-datasets endpoint is rate-limited, and users have an explicit refresh button when
 * they want fresh data. Create/delete mutations invalidate the key directly, and the
 * refresh `refetch()` bypasses `staleTime`, so the cache stays correct after writes.
 */
export const useDatasetsPageQuery = (params: UseDatasetsPageQueryParams) => {
  return useQuery({
    queryKey: [V2_DATASETS_PAGE_QUERY_KEY, params.experimentId, params.nameFilter, params.pageSize, params.pageToken],
    queryFn: () => fetchDatasetsPage(params),
    // 5 min cache window keeps the data warm across short navigations; `staleTime: Infinity`
    // prevents auto-refetch on revisit so we don't burn the rate-limited list endpoint.
    cacheTime: 5 * 60 * 1000,
    staleTime: Infinity,
    keepPreviousData: true,
    refetchOnWindowFocus: false,
    retry: false,
  });
};
