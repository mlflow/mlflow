import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { LabelSchema } from '../types';

export const GET_LABEL_SCHEMA_QUERY_KEY = 'GET_LABEL_SCHEMA';

interface GetLabelSchemaResponse {
  label_schema: LabelSchema;
}

/**
 * Fetch a single label schema by its server-generated `schema_id`.
 *
 * Identity for tracking-store schemas is `(experiment_id, name)` but the
 * canonical primary key on the wire is `schema_id`. Use
 * `useGetLabelSchemaByNameQuery` for the experiment-scoped lookup.
 */
export const useGetLabelSchemaQuery = ({ schemaId, enabled = true }: { schemaId: string; enabled?: boolean }) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<GetLabelSchemaResponse, Error>({
    queryKey: [GET_LABEL_SCHEMA_QUERY_KEY, schemaId],
    queryFn: async () => {
      const params = new URLSearchParams({ schema_id: schemaId });
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/label-schemas/get?${params.toString()}`), {
        method: 'GET',
      })) as GetLabelSchemaResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(schemaId),
  });

  return {
    labelSchema: data?.label_schema,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
