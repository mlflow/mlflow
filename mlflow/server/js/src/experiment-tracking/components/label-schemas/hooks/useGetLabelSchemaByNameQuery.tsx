import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import type { LabelSchema } from '../types';

export const GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY = 'GET_LABEL_SCHEMA_BY_NAME';

interface GetLabelSchemaByNameResponse {
  label_schema: LabelSchema;
}

/**
 * Fetch a label schema by `(experiment_id, name)`.
 *
 * The server uses `schema_id` as the canonical primary key but
 * `(experiment_id, name)` is unique and recoverable client-side from
 * the user-supplied schema name, so this is the typical lookup path
 * for "does a schema with this name already exist on this experiment?".
 */
export const useGetLabelSchemaByNameQuery = ({
  experimentId,
  name,
  enabled = true,
}: {
  experimentId: string;
  name: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<GetLabelSchemaByNameResponse, Error>({
    queryKey: [GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY, experimentId, name],
    queryFn: async () => {
      const params = new URLSearchParams({ experiment_id: experimentId, name });
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/label-schemas/get-by-name?${params.toString()}`), {
        method: 'GET',
      })) as GetLabelSchemaByNameResponse;
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(experimentId) && Boolean(name),
  });

  return {
    labelSchema: data?.label_schema,
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
