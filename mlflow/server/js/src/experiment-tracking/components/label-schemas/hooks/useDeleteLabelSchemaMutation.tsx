import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { GET_LABEL_SCHEMA_QUERY_KEY } from './useGetLabelSchemaQuery';
import { GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY } from './useGetLabelSchemaByNameQuery';
import { LIST_LABEL_SCHEMAS_QUERY_KEY } from './useListLabelSchemasQuery';
import { LABEL_SCHEMAS_API_BASE } from './constants';

export interface DeleteLabelSchemaParams {
  schema_id: string;
}

/**
 * Delete a label schema. The server treats missing schemas as no-ops
 * (returns 200 with an empty body), so callers don't need to special-case
 * the "schema was already gone" condition.
 */
export const useDeleteLabelSchemaMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<void, Error, DeleteLabelSchemaParams>({
    mutationFn: async ({ schema_id }) => {
      await fetchAPI(getAjaxUrl(`${LABEL_SCHEMAS_API_BASE}/delete`), {
        method: 'DELETE',
        body: { schema_id },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_LABEL_SCHEMAS_QUERY_KEY]);
      queryClient.invalidateQueries([GET_LABEL_SCHEMA_QUERY_KEY]);
      queryClient.invalidateQueries([GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY]);
    },
  });

  return {
    deleteLabelSchema: mutate,
    deleteLabelSchemaAsync: mutateAsync,
    isDeleting: isLoading,
    error,
  };
};
