import { useCallback, useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@databricks/web-shared/query-client';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';
import {
  listDatasetRecordsQueryKey,
  useCreateDatasetRecordMutation,
} from '../hooks/useDatasetsQueries';
import type { SaveStatus } from '../components/DatasetRecordDetailFooter';
import { getDefaultRecord, validateSchemaConsistency } from '../utils/datasetSchemaUtils';
import { useDatasetRecordEditorState } from './useDatasetRecordEditorState';

/**
 * Snapshot of the in-progress "new record" create form. Mirrored to the records table so the
 * phantom row reflects what the user is typing in real time. `*Text` carries the raw editor
 * contents (so the row updates even when the JSON is mid-edit); the parsed objects are set
 * only when valid.
 */
export interface PendingNewRecord {
  inputsText: string;
  expectationsText: string;
  inputs: Record<string, unknown> | undefined;
  expectations: Record<string, unknown> | undefined;
  tags: Record<string, string>;
}

interface UseRecordCreateStateParams {
  datasetId: string;
  /** Localized fallback used when a create error isn't an Error instance. */
  fallbackErrorMessage: string;
  /**
   * Current full set of records. Gated by `validateSchemaConsistency` so a mixed
   * singleturn/multiturn dataset can't be introduced by the new record.
   */
  existingRecords: DatasetRecord[];
  onSaveSuccess?: () => void;
  onSaveError?: (error: Error) => void;
  /** Live preview hook for the phantom row in the records table. Fires on every editor change. */
  onPendingChange?: (next: PendingNewRecord) => void;
}

interface UseRecordCreateStateResult {
  inputs: ReturnType<typeof useDatasetRecordEditorState>;
  expectations: ReturnType<typeof useDatasetRecordEditorState>;
  tags: Record<string, string>;
  setTags: (next: Record<string, string>) => void;
  status: SaveStatus;
  errorMessage: string | undefined;
  isDirty: boolean;
  hasContent: boolean;
  save: () => void;
  discard: () => void;
  onContainerKeyDown: (event: React.KeyboardEvent) => void;
}

/**
 * Sibling of `useRecordSaveState` for the "new record" create flow. Uses the create-mutation
 * RPC (vs upsert), starts with an empty baseline (vs the source record), and emits live
 * preview events so the side panel can show a synthetic row in the records table that
 * reflects what the user is typing.
 */
export const useRecordCreateState = ({
  datasetId,
  fallbackErrorMessage,
  existingRecords,
  onSaveSuccess,
  onSaveError,
  onPendingChange,
}: UseRecordCreateStateParams): UseRecordCreateStateResult => {
  // Seed the editors with v1's default record shape (singleturn `messages` for empty/messages-
  // shaped datasets, multiturn `goal`/`persona` when existing rows already use a `goal` key) so
  // the user can tweak rather than type from scratch. Lazy `useState` locks the snapshot to
  // mount time — a peer-tab refetch of `existingRecords` must not swap defaults under the user
  // mid-edit. baseline == seeded, so `isDirty` is false on first paint; the save-FSM keys off
  // `hasContent` instead so the user can still submit the unedited seed.
  const [seededDefaults] = useState(() => getDefaultRecord(existingRecords));
  // recordId: undefined keeps editor state stuck to its initial value for the lifetime
  // of the panel — opening the panel a second time after a successful save starts fresh
  // because the panel unmounts/remounts (orchestrated by the page).
  const inputs = useDatasetRecordEditorState({ recordId: undefined, initialValue: seededDefaults.inputs });
  const expectations = useDatasetRecordEditorState({
    recordId: undefined,
    initialValue: seededDefaults.expectations,
  });
  // Draft tags live entirely in local state — no upsert mutation here because there's no
  // record ID until the create RPC resolves. Tags are submitted alongside inputs/expectations
  // in the same create call, then this state is reset on success.
  const [tags, setTags] = useState<Record<string, string>>({});

  const createMutation = useCreateDatasetRecordMutation(datasetId);
  const queryClient = useQueryClient();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [justSaved, setJustSaved] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);

  const hasTags = Object.keys(tags).length > 0;
  const isDirty = inputs.isDirty || expectations.isDirty || hasTags;
  const hasContent = inputs.text.trim() !== '' || expectations.text.trim() !== '' || hasTags;
  const anyInvalid = !inputs.isValid || !expectations.isValid;
  // Mirrors the create-mode gate from the old AddRecordModal: backend rejects records with
  // empty inputs, so disable Save until the user provides at least one key.
  const inputsHasData = inputs.parsed !== undefined && Object.keys(inputs.parsed).length > 0;
  // Keyed off `hasContent` (not `isDirty`) so typing `{}` on top of the seeded defaults still
  // surfaces the empty-inputs gate even though baseline already happens to be non-empty.
  const inputsEmpty = hasContent && !inputsHasData;

  const status: SaveStatus = useMemo(() => {
    if (createMutation.isLoading || isVerifying) return 'saving';
    if (errorMessage) return 'error';
    if (anyInvalid) return 'invalid';
    if (inputsEmpty) return 'empty-inputs';
    // `justSaved` is checked before `hasContent` so a successful save resolves to 'saved'
    // rather than the savable-from-seeded 'dirty' branch below (the post-save baseline equals
    // the just-typed text, so hasContent is still true).
    if (justSaved) return 'saved';
    // `hasContent` rather than `isDirty`: seeded defaults make the form savable from mount
    // even though baseline == text. Save still requires valid + non-empty inputs (gated above).
    if (hasContent) return 'dirty';
    return 'clean';
  }, [createMutation.isLoading, isVerifying, errorMessage, anyInvalid, inputsEmpty, hasContent, justSaved]);

  // Stop saying "saved" the moment the user edits again.
  useEffect(() => {
    if (isDirty) setJustSaved(false);
  }, [isDirty]);

  // Emit a preview pulse to the fake row whenever the editor changes. Carry the raw text so
  // partial/invalid JSON still shows up live in the row; carry the parsed object too so the
  // row can prefer compact `JSON.stringify` output over a multi-line pretty-printed source.
  useEffect(() => {
    if (!onPendingChange) return;
    onPendingChange({
      inputsText: inputs.text,
      expectationsText: expectations.text,
      inputs: inputs.isValid ? inputs.parsed : undefined,
      expectations: expectations.isValid ? expectations.parsed : undefined,
      tags,
    });
  }, [
    inputs.text,
    inputs.isValid,
    inputs.parsed,
    expectations.text,
    expectations.isValid,
    expectations.parsed,
    tags,
    onPendingChange,
  ]);

  const save = useCallback(() => {
    if (!hasContent || anyInvalid || !inputsHasData) return;
    setErrorMessage(undefined);

    const syntheticNew = {
      dataset_record_id: '__new__',
      inputs: inputs.parsed ?? {},
      expectations: expectations.parsed,
    } as DatasetRecord;

    // Refetch records before validating so a peer-tab create-of-different-schema can't slip
    // past the gate. Tolerate a refetch failure by falling back to in-memory existingRecords.
    setIsVerifying(true);
    const recordsKey = listDatasetRecordsQueryKey(datasetId);
    queryClient
      .refetchQueries({ queryKey: recordsKey })
      .catch(() => undefined)
      .then(() => {
        const freshRecords = queryClient.getQueryData<DatasetRecord[]>(recordsKey) ?? existingRecords;

        try {
          validateSchemaConsistency([...freshRecords, syntheticNew]);
        } catch (validationErr) {
          const error = validationErr instanceof Error ? validationErr : new Error(fallbackErrorMessage);
          setIsVerifying(false);
          setErrorMessage(error.message);
          onSaveError?.(error);
          return;
        }

        setIsVerifying(false);
        createMutation.mutate(
          {
            inputs: inputs.parsed,
            expectations: expectations.parsed,
            // Skip the field when empty so the request body stays minimal — the backend
            // tolerates `tags: {}` but the listing cache produces noisier diffs.
            ...(hasTags ? { tags } : {}),
          },
          {
            onSuccess: () => {
              // Advance the baseline to the just-typed text so isDirty becomes false; the
              // FSM then resolves to 'saved'. The parent typically closes the panel right
              // after, but resetting also keeps the status FSM honest if it stays open.
              inputs.reset(inputs.text);
              expectations.reset(expectations.text);
              // Reset tags too — otherwise hasTags keeps isDirty true and the FSM lands
              // on 'dirty' instead of 'saved'.
              setTags({});
              setJustSaved(true);
              onSaveSuccess?.();
            },
            onError: (err) => {
              const error = err instanceof Error ? err : new Error(fallbackErrorMessage);
              setErrorMessage(error.message);
              onSaveError?.(error);
            },
          },
        );
      });
  }, [
    anyInvalid,
    createMutation,
    datasetId,
    existingRecords,
    expectations,
    fallbackErrorMessage,
    hasContent,
    hasTags,
    inputs,
    inputsHasData,
    onSaveError,
    onSaveSuccess,
    queryClient,
    tags,
  ]);

  const discard = useCallback(() => {
    inputs.reset();
    expectations.reset();
    setTags({});
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

  return {
    inputs,
    expectations,
    tags,
    setTags,
    status,
    errorMessage,
    isDirty,
    hasContent,
    save,
    discard,
    onContainerKeyDown,
  };
};
