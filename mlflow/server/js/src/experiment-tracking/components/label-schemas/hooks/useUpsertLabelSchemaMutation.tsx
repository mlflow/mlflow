import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { GET_LABEL_SCHEMA_QUERY_KEY } from './useGetLabelSchemaQuery';
import { GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY } from './useGetLabelSchemaByNameQuery';
import { LIST_LABEL_SCHEMAS_QUERY_KEY } from './useListLabelSchemasQuery';
import type { LabelSchema, LabelSchemaInput, LabelSchemaType } from '../types';

export interface UpsertLabelSchemaParams {
  experiment_id: string;
  name: string;
  type: LabelSchemaType;
  title: string;
  input: LabelSchemaInput;
  instruction?: string;
  /**
   * `enable_comment` is intentionally optional: omitting it preserves
   * the existing value on replace (and defaults to `false` on create).
   * Pass `true` or `false` to set explicitly.
   */
  enable_comment?: boolean;
}

interface UpsertLabelSchemaResponse {
  label_schema: LabelSchema;
}

/**
 * Atomically create-or-replace by `(experiment_id, name)`. `type` is
 * immutable on replace; a type mismatch with the existing row is
 * rejected with `INVALID_PARAMETER_VALUE`.
 */
export const useUpsertLabelSchemaMutation = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error, reset } = useMutation<
    UpsertLabelSchemaResponse,
    Error,
    UpsertLabelSchemaParams
  >({
    mutationFn: async (params) => {
      // Strip undefined keys so the server's HasField gates preserve
      // "absent on the wire = unchanged on replace" semantics for
      // `enable_comment` and `instruction`.
      const body: UpsertLabelSchemaParams = {
        experiment_id: params.experiment_id,
        name: params.name,
        type: params.type,
        title: params.title,
        input: params.input,
      };
      if (params.instruction !== undefined) {
        body.instruction = params.instruction;
      }
      if (params.enable_comment !== undefined) {
        body.enable_comment = params.enable_comment;
      }
      return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/label-schemas/upsert'), {
        method: 'POST',
        body,
      })) as UpsertLabelSchemaResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_LABEL_SCHEMAS_QUERY_KEY]);
      queryClient.invalidateQueries([GET_LABEL_SCHEMA_QUERY_KEY]);
      queryClient.invalidateQueries([GET_LABEL_SCHEMA_BY_NAME_QUERY_KEY]);
    },
  });

  return {
    upsertLabelSchema: mutate,
    upsertLabelSchemaAsync: mutateAsync,
    isUpserting: isLoading,
    error,
    reset,
  };
};
