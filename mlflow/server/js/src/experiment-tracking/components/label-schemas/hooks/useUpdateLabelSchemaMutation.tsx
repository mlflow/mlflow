import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { GET_LABEL_SCHEMA_QUERY_KEY } from './useGetLabelSchemaQuery';
import { GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY } from './useGetLabelSchemaByNameQuery';
import { LIST_LABEL_SCHEMAS_QUERY_KEY } from './useListLabelSchemasQuery';
import type { LabelSchema, LabelSchemaInput } from '../types';

export interface UpdateLabelSchemaParams {
  schema_id: string;
  /**
   * Sparse update: only fields explicitly set are forwarded. Fields left
   * undefined are unchanged on the server. Empty strings are real values
   * (they replace the stored field with `""`), not no-ops; pass `undefined`
   * (omit the key) to leave a field unchanged.
   *
   * `type` is immutable post-create and is not on this surface.
   */
  name?: string;
  title?: string;
  instruction?: string;
  enable_comment?: boolean;
  input?: LabelSchemaInput;
}

interface UpdateLabelSchemaResponse {
  label_schema: LabelSchema;
}

export const useUpdateLabelSchemaMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    UpdateLabelSchemaResponse,
    Error,
    UpdateLabelSchemaParams
  >({
    mutationFn: async (params) => {
      // Strip undefined keys before sending so HasField semantics on the
      // server distinguish "user didn't say" from "user set null/empty".
      const body: UpdateLabelSchemaParams = { schema_id: params.schema_id };
      if (params.name !== undefined) {
        body.name = params.name;
      }
      if (params.title !== undefined) {
        body.title = params.title;
      }
      if (params.instruction !== undefined) {
        body.instruction = params.instruction;
      }
      if (params.enable_comment !== undefined) {
        body.enable_comment = params.enable_comment;
      }
      if (params.input !== undefined) {
        body.input = params.input;
      }
      return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/label-schemas/update'), {
        method: 'PATCH',
        body,
      })) as UpdateLabelSchemaResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_LABEL_SCHEMAS_QUERY_KEY]);
      queryClient.invalidateQueries([GET_LABEL_SCHEMA_QUERY_KEY]);
      queryClient.invalidateQueries([GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY]);
    },
  });

  return {
    updateLabelSchema: mutate,
    updateLabelSchemaAsync: mutateAsync,
    isUpdating: isLoading,
    error,
  };
};
