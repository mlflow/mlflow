import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY } from './useGetLabelSchemaByNameQuery';
import { LIST_LABEL_SCHEMAS_QUERY_KEY } from './useListLabelSchemasQuery';
import type { LabelSchema, LabelSchemaInput, LabelSchemaType } from '../types';
import { LABEL_SCHEMAS_API_BASE } from './constants';

export interface CreateLabelSchemaParams {
  experiment_id: string;
  name: string;
  type: LabelSchemaType;
  input: LabelSchemaInput;
  instruction?: string;
  enable_comment?: boolean;
}

interface CreateLabelSchemaResponse {
  label_schema: LabelSchema;
}

/**
 * Create a new label schema. The server rejects a `(experiment_id, name)`
 * collision with `RESOURCE_ALREADY_EXISTS`; use the update mutation to
 * edit an existing schema.
 *
 * On success, invalidates the list + by-name query caches so any open
 * admin view picks up the new row.
 */
export const useCreateLabelSchemaMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error, reset } = useMutation<
    CreateLabelSchemaResponse,
    Error,
    CreateLabelSchemaParams
  >({
    mutationFn: async (params) => {
      return (await fetchAPI(getAjaxUrl(`${LABEL_SCHEMAS_API_BASE}/create`), {
        method: 'POST',
        body: params,
      })) as CreateLabelSchemaResponse;
    },
    onSuccess: () => {
      // GET-by-id is intentionally not invalidated: a newly-created schema
      // has a server-generated schema_id the caller didn't know before, so
      // no existing GET cache entry can be stale. Update / upsert / delete
      // all invalidate GET because they target a known schema_id.
      queryClient.invalidateQueries([LIST_LABEL_SCHEMAS_QUERY_KEY]);
      queryClient.invalidateQueries([GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY]);
    },
  });

  return {
    createLabelSchema: mutate,
    createLabelSchemaAsync: mutateAsync,
    isCreating: isLoading,
    error,
    reset,
  };
};
