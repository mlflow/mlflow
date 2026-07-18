import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { LabelSchema } from '../types';
import { LABEL_SCHEMAS_API_BASE } from './constants';

export const LIST_LABEL_SCHEMAS_QUERY_KEY = 'LIST_LABEL_SCHEMAS';

interface ListLabelSchemasResponse {
  label_schemas?: LabelSchema[];
  next_page_token?: string;
}

/**
 * Paginated list of label schemas for an experiment, ordered by
 * `created_at desc` (server-side; see `SqlAlchemyStore.list_label_schemas`).
 *
 * `maxResults` defaults to the server-side default (100); the handler
 * rejects values outside `[1, SEARCH_MAX_RESULTS_THRESHOLD]`.
 */
export const useListLabelSchemasQuery = ({
  experimentId,
  maxResults,
  pageToken,
  enabled = true,
}: {
  experimentId: string;
  maxResults?: number;
  pageToken?: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<ListLabelSchemasResponse, Error>({
    queryKey: [LIST_LABEL_SCHEMAS_QUERY_KEY, experimentId, maxResults, pageToken],
    queryFn: async () => {
      const params = new URLSearchParams({ experiment_id: experimentId });
      // Guard out 0/negative client-side; the handler enforces
      // max_results in [1, SEARCH_MAX_RESULTS_THRESHOLD] and would
      // otherwise return INVALID_PARAMETER_VALUE for a 0.
      if (maxResults != null && maxResults > 0) {
        params.set('max_results', String(maxResults));
      }
      if (pageToken) {
        params.set('page_token', pageToken);
      }
      return (await fetchAPI(getAjaxUrl(`${LABEL_SCHEMAS_API_BASE}/list?${params.toString()}`), {
        method: 'GET',
      })) as ListLabelSchemasResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(experimentId),
  });

  return {
    labelSchemas: data?.label_schemas ?? [],
    nextPageToken: data?.next_page_token,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
