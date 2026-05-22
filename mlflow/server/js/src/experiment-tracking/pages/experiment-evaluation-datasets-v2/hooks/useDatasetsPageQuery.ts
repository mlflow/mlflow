import { useQuery } from '@databricks/web-shared/query-client';
import { workspaceFetch } from '@databricks/web-shared/spog/workspace-console';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { Dataset } from '../hooks/useDatasetsQueries';
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

/**
 * Escapes a single quote for embedding inside an `experiment_id='X'`-style filter.
 * Backend uses SQL-style filter syntax; quoting prevents injection from user-typed search.
 * No spaces around `=`: the managed-evals RPC pre-validator (QueryConstraint.isValidQuery)
 * anchors a regex that rejects whitespace before `=`, so this output must stay tight.
 */
const escapeFilterLiteral = (value: string) => value.replace(/'/g, "''");

/**
 * Strips SQL LIKE wildcards (`%` matches any run; `_` matches one char) from user-typed
 * search input. Without this, a search for `100%` becomes a true wildcard match and a search
 * for `data_set_1` lets `_` match any single character — both surprise the user.
 *
 * Stripping (rather than escaping with `ESCAPE 'X'`) is forced by the backend: the managed-
 * evals filter grammar (`managed-evals/src/common/DatasetSearchFilter.scala`) recognizes only
 * `=`, `!=`, `LIKE`, `ILIKE` and would reject any `ESCAPE` clause we tried to send. Dataset
 * names almost never contain literal `%` or `_`, so the lost fidelity is acceptable.
 */
const stripLikeWildcards = (value: string) => value.replace(/[%_]/g, '');

const buildFilterString = (experimentId: string, nameFilter: string) => {
  const clauses = [`experiment_id='${escapeFilterLiteral(experimentId)}'`];
  const sanitized = stripLikeWildcards(nameFilter.trim());
  if (sanitized) {
    clauses.push(`name ILIKE '%${escapeFilterLiteral(sanitized)}%'`);
  }
  return clauses.join(' AND ');
};

const fetchDatasetsPage = async ({
  experimentId,
  nameFilter,
  pageSize,
  pageToken,
}: UseDatasetsPageQueryParams): Promise<ListDatasetsPageResponse> => {
  const params = new URLSearchParams({
    filter: buildFilterString(experimentId, nameFilter),
    page_size: String(pageSize),
    order_by: DEFAULT_DATASETS_ORDER_BY,
  });
  if (pageToken) {
    params.set('page_token', pageToken);
  }

  const res = await workspaceFetch(`${getAjaxUrl('ajax-api/2.0/managed-evals/datasets')}?${params}`);
  if (!res.ok) {
    let message = `Failed to fetch datasets: ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.message) {
        message = body.message;
      }
    } catch {
      // body wasn't JSON; fall back to the status line.
    }
    throw new Error(message);
  }

  return res.json();
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
