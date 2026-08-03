import { useCallback, useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@databricks/web-shared/query-client';
import type { DatasetRecord } from './useDatasetsQueries';
import { listDatasetRecordsQueryKey, useUpsertDatasetRecordsMutation } from './useDatasetsQueries';
import type { SaveStatus } from '../components/DatasetRecordDetailFooter';
import { validateSchemaConsistency } from '../utils/datasetSchemaUtils';
import { useDatasetRecordEditorState } from './useDatasetRecordEditorState';

interface UseRecordSaveStateParams {
  datasetId: string;
  record: DatasetRecord | undefined;
  /** Localized fallback used when a save error isn't an Error instance. */
  fallbackErrorMessage: string;
  /**
   * The current full set of records in this dataset. Used to gate the save with
   * `validateSchemaConsistency` so a user can't introduce a mixed singleturn/multiturn
   * schema (the evaluation runner rejects such datasets downstream).
   */
  existingRecords: DatasetRecord[];
  onSaveSuccess?: () => void;
  onSaveError?: (error: Error) => void;
}

interface UseRecordSaveStateResult {
  inputs: ReturnType<typeof useDatasetRecordEditorState>;
  expectations: ReturnType<typeof useDatasetRecordEditorState>;
  status: SaveStatus;
  errorMessage: string | undefined;
  isDirty: boolean;
  /**
   * True when either editor has non-whitespace text. Distinct from `isDirty`: this is the
   * signal the side panel's discard-guard uses to decide whether closing should silently
   * drop state (`hasContent === false`) or prompt the user (`hasContent === true`).
   */
  hasContent: boolean;
  save: () => void;
  discard: () => void;
  /** Cmd/Ctrl-S save handler. Attach to the drawer container so the listener stays scoped. */
  onContainerKeyDown: (event: React.KeyboardEvent) => void;
}

/**
 * Owns the JSON editor state for inputs/expectations plus the save FSM for the records drawer.
 * Lifts the orchestration out of the drawer component so the drawer can focus on layout.
 */
export const useRecordSaveState = ({
  datasetId,
  record,
  fallbackErrorMessage,
  existingRecords,
  onSaveSuccess,
  onSaveError,
}: UseRecordSaveStateParams): UseRecordSaveStateResult => {
  const inputs = useDatasetRecordEditorState({
    recordId: record?.dataset_record_id,
    initialValue: record?.inputs,
  });
  const expectations = useDatasetRecordEditorState({
    recordId: record?.dataset_record_id,
    initialValue: record?.expectations,
  });

  const upsertMutation = useUpsertDatasetRecordsMutation(datasetId);
  const queryClient = useQueryClient();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [justSaved, setJustSaved] = useState(false);
  // True from the moment the user clicks Save until the pre-save schema refetch + validate
  // step resolves. Merged into the visible 'saving' status so the button shows a spinner
  // immediately and the user gets feedback even when the refetch is slow.
  const [isVerifying, setIsVerifying] = useState(false);

  const isDirty = inputs.isDirty || expectations.isDirty;
  const hasContent = inputs.text.trim() !== '' || expectations.text.trim() !== '';
  const anyInvalid = !inputs.isValid || !expectations.isValid;
  // Guard against the empty-editor → empty-object → server-wipes-field data-loss path.
  // `parseRecordObject('')` parses to `{}`, which `transformDatasetRecordForUpdate` turns into
  // `inputs: []`; with `update_mask=inputs` the server then clears the field. Expectations
  // are legitimately optional and may remain empty — only inputs are gated here, matching
  // the create-flow rule that inputs must have at least one key.
  const inputsEmpty = inputs.parsed !== undefined && Object.keys(inputs.parsed).length === 0;

  const status: SaveStatus = useMemo(() => {
    if (upsertMutation.isLoading || isVerifying) return 'saving';
    if (errorMessage) return 'error';
    if (anyInvalid) return 'invalid';
    if (inputsEmpty && isDirty) return 'empty-inputs';
    if (isDirty) return 'dirty';
    if (justSaved) return 'saved';
    return 'clean';
  }, [upsertMutation.isLoading, isVerifying, errorMessage, anyInvalid, inputsEmpty, isDirty, justSaved]);

  // Stop saying "saved" the moment the user edits again.
  useEffect(() => {
    if (isDirty) setJustSaved(false);
  }, [isDirty]);

  // Clear save banners whenever the drawer switches to a different record.
  useEffect(() => {
    setErrorMessage(undefined);
    setJustSaved(false);
  }, [record?.dataset_record_id]);

  const save = useCallback(() => {
    if (!record || !isDirty || anyInvalid || inputsEmpty) return;
    setErrorMessage(undefined);

    // `parsed` is always defined for any dirty field — anyInvalid + inputsEmpty guards above
    // would have returned otherwise. `updates` and `updateMask` carry the same partial: the
    // mask tells the backend which fields to write, `updates` carries the values.
    const edits: Partial<DatasetRecord> = {};
    if (inputs.isDirty) edits.inputs = inputs.parsed;
    if (expectations.isDirty) edits.expectations = expectations.parsed;

    // Capture the text being submitted so onSuccess can advance the baseline to exactly what
    // we just sent — independent of when the optimistic-cache re-render flushes the ref.
    const submittedInputsText = inputs.text;
    const submittedExpectationsText = expectations.text;

    // Refetch the records list before running schema validation. Without this, a peer-tab
    // edit (or an in-tab create) that hasn't yet propagated to this hook's `existingRecords`
    // prop could let a mixed singleturn/multiturn save sneak past the client-side gate.
    // We tolerate refetch failure: if the network is down we fall back to the in-memory
    // `existingRecords` and let the mutation surface the server's own validation error.
    setIsVerifying(true);
    const recordsKey = listDatasetRecordsQueryKey(datasetId);
    queryClient
      .refetchQueries({ queryKey: recordsKey })
      .catch(() => undefined)
      .then(() => {
        const freshRecords = queryClient.getQueryData<DatasetRecord[]>(recordsKey) ?? existingRecords;

        try {
          validateSchemaConsistency(freshRecords, { [record.dataset_record_id]: edits });
        } catch (validationErr) {
          const error = validationErr instanceof Error ? validationErr : new Error(fallbackErrorMessage);
          setIsVerifying(false);
          setErrorMessage(error.message);
          onSaveError?.(error);
          return;
        }

        setIsVerifying(false);
        upsertMutation.mutate([{ recordId: record.dataset_record_id, updates: edits, updateMask: edits }], {
          onSuccess: () => {
            inputs.reset(submittedInputsText);
            expectations.reset(submittedExpectationsText);
            setJustSaved(true);
            onSaveSuccess?.();
          },
          onError: (err) => {
            const error = err instanceof Error ? err : new Error(fallbackErrorMessage);
            setErrorMessage(error.message);
            onSaveError?.(error);
          },
        });
      });
  }, [
    anyInvalid,
    datasetId,
    existingRecords,
    expectations,
    fallbackErrorMessage,
    inputs,
    inputsEmpty,
    isDirty,
    onSaveError,
    onSaveSuccess,
    queryClient,
    record,
    upsertMutation,
  ]);

  const discard = useCallback(() => {
    inputs.reset();
    expectations.reset();
    setErrorMessage(undefined);
  }, [inputs, expectations]);

  const onContainerKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 's') {
        event.preventDefault();
        save();
      }
    },
    [save],
  );

  return { inputs, expectations, status, errorMessage, isDirty, hasContent, save, discard, onContainerKeyDown };
};
