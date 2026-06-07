import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDebouncedCallback } from 'use-debounce';
import { useQueryClient } from '@databricks/web-shared/query-client';
import type { DatasetRecord } from './useDatasetsQueries';
import { listDatasetRecordsQueryKey, useUpdateDatasetRecordMutation } from './useDatasetsQueries';
import type { SaveStatus } from '../components/DatasetRecordDetailFooter';
import { validateSchemaConsistency } from '../utils/datasetSchemaUtils';
import { useDatasetRecordEditorState } from './useDatasetRecordEditorState';

/**
 * How edits are committed:
 * - `autosave` (OSS default): a valid, non-empty edit is persisted on a debounce; no button.
 * - `explicit`: nothing is persisted until `save()` is called (e.g. a "Save" button). The
 *   Databricks/UC port flips to this because UC writes aren't trivially undoable. Both modes
 *   share the same commit + validation plumbing below.
 */
export type CommitMode = 'autosave' | 'explicit';

/** Debounce window for autosave (ms). Long enough to coalesce a burst of keystrokes. */
const DEFAULT_AUTOSAVE_DEBOUNCE_MS = 800;

/** Stable key for an edit's content, used to suppress re-saving a value that just failed. */
const signatureOf = (inputsText: string, expectationsText: string) => `${inputsText}\u0000${expectationsText}`;

/** Everything a single commit needs, captured at schedule time so a flush-on-switch replays
 * the *outgoing* record's edit rather than whatever the editor happens to hold post-switch. */
interface CommitPayload {
  recordId: string;
  edits: Partial<DatasetRecord>;
  inputsText: string;
  expectationsText: string;
}

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
  onSaveError?: (error: Error) => void;
  /** Commit trigger. Defaults to `autosave`. */
  commitMode?: CommitMode;
  /** Autosave debounce window (ms). Defaults to {@link DEFAULT_AUTOSAVE_DEBOUNCE_MS}. */
  debounceMs?: number;
}

interface UseRecordSaveStateResult {
  inputs: ReturnType<typeof useDatasetRecordEditorState>;
  expectations: ReturnType<typeof useDatasetRecordEditorState>;
  status: SaveStatus;
  errorMessage: string | undefined;
  isDirty: boolean;
  /** True when either editor has non-whitespace text. */
  hasContent: boolean;
  /** Commit any pending edit immediately (cancels the debounce). Used by Cmd/Ctrl-S and the
   * explicit-mode Save button. */
  save: () => void;
  /** Synchronously commit a pending autosave — call before a record-switch / close so the
   * outgoing edit isn't dropped. No-op when nothing is pending. */
  flush: () => void;
  /** Revert both editors to the last-saved record values and drop any pending autosave. */
  discard: () => void;
  /** Cmd/Ctrl-S handler. Attach to the drawer container so the listener stays scoped. */
  onContainerKeyDown: (event: React.KeyboardEvent) => void;
}

/**
 * Owns the JSON editor state for inputs/expectations plus the save lifecycle for an existing
 * dataset record. In `autosave` mode a valid, non-empty edit is committed on a debounce via the
 * update-by-id endpoint (which, unlike upsert, can change `inputs` in place without orphaning
 * the row) — so there is no separate "Save" step. Invalid or empty-inputs edits are *deferred*:
 * they are never persisted, the last valid value stays on the server, and the footer surfaces
 * why. A pending edit is flushed on record-switch / unmount so it is never silently lost.
 */
export const useRecordSaveState = ({
  datasetId,
  record,
  fallbackErrorMessage,
  existingRecords,
  onSaveError,
  commitMode = 'autosave',
  debounceMs = DEFAULT_AUTOSAVE_DEBOUNCE_MS,
}: UseRecordSaveStateParams): UseRecordSaveStateResult => {
  const inputs = useDatasetRecordEditorState({
    recordId: record?.dataset_record_id,
    initialValue: record?.inputs,
  });
  const expectations = useDatasetRecordEditorState({
    recordId: record?.dataset_record_id,
    initialValue: record?.expectations,
  });

  const updateMutation = useUpdateDatasetRecordMutation(datasetId);
  const queryClient = useQueryClient();
  const [errorMessage, setErrorMessage] = useState<string | undefined>();
  const [justSaved, setJustSaved] = useState(false);
  // Signature of the last edit that failed to save. While the editor still holds exactly that
  // value we don't re-attempt — otherwise a persistent error (e.g. a duplicate-inputs conflict)
  // would have the debounce re-fire the identical doomed request every interval and spam errors.
  // Editing to any other value clears the block (the signature stops matching).
  const [failedSignature, setFailedSignature] = useState<string | null>(null);
  // True from the moment a commit starts until its pre-save schema refetch + validate resolves,
  // merged into the visible 'saving' status so feedback is immediate even on a slow refetch.
  const [isVerifying, setIsVerifying] = useState(false);

  const isDirty = inputs.isDirty || expectations.isDirty;
  const hasContent = inputs.text.trim() !== '' || expectations.text.trim() !== '';
  const anyInvalid = !inputs.isValid || !expectations.isValid;
  // Empty inputs would round-trip as `inputs: {}` and wipe the field; gate it (expectations may
  // legitimately be empty). Matches the create-flow rule that inputs need at least one key.
  const inputsEmpty = inputs.parsed !== undefined && Object.keys(inputs.parsed).length === 0;
  // A commit is only allowed when there is a dirty, valid, non-empty-inputs edit on a real record.
  const canCommit = Boolean(record) && isDirty && !anyInvalid && !inputsEmpty;

  // The actual write. Takes an explicit payload (not closure state) so a flush during a
  // record-switch commits the record that was being edited, not the one being switched to.
  const commit = useCallback(
    (payload: CommitPayload) => {
      setErrorMessage(undefined);
      setIsVerifying(true);
      const recordsKey = listDatasetRecordsQueryKey(datasetId);
      queryClient
        .refetchQueries({ queryKey: recordsKey })
        .catch(() => undefined)
        .then(() => {
          const freshRecords = queryClient.getQueryData<DatasetRecord[]>(recordsKey) ?? existingRecords;
          const signature = signatureOf(payload.inputsText, payload.expectationsText);
          try {
            validateSchemaConsistency(freshRecords, { [payload.recordId]: payload.edits });
          } catch (validationErr) {
            const error = validationErr instanceof Error ? validationErr : new Error(fallbackErrorMessage);
            setIsVerifying(false);
            setErrorMessage(error.message);
            setFailedSignature(signature);
            onSaveError?.(error);
            return;
          }

          setIsVerifying(false);
          updateMutation.mutate([{ recordId: payload.recordId, updates: payload.edits }], {
            onSuccess: () => {
              // Advance each editor's baseline to exactly what we submitted so isDirty clears
              // without waiting for the optimistic cache re-render to flush.
              inputs.reset(payload.inputsText);
              expectations.reset(payload.expectationsText);
              setFailedSignature(null);
              setJustSaved(true);
            },
            onError: (err) => {
              const error = err instanceof Error ? err : new Error(fallbackErrorMessage);
              setErrorMessage(error.message);
              // Block re-firing this exact value; editing to anything else clears it.
              setFailedSignature(signature);
              onSaveError?.(error);
            },
          });
        });
    },
    [datasetId, existingRecords, fallbackErrorMessage, inputs, expectations, onSaveError, queryClient, updateMutation],
  );

  const debouncedCommit = useDebouncedCallback(commit, debounceMs);

  // Build the payload from the *current* editor state. `edits` carries only the dirty fields.
  const buildPayload = useCallback((): CommitPayload | null => {
    if (!record) return null;
    const edits: Partial<DatasetRecord> = {};
    if (inputs.isDirty) edits.inputs = inputs.parsed;
    if (expectations.isDirty) edits.expectations = expectations.parsed;
    return {
      recordId: record.dataset_record_id,
      edits,
      inputsText: inputs.text,
      expectationsText: expectations.text,
    };
  }, [
    record,
    inputs.isDirty,
    inputs.parsed,
    inputs.text,
    expectations.isDirty,
    expectations.parsed,
    expectations.text,
  ]);

  const currentSignature = signatureOf(inputs.text, expectations.text);

  // Autosave trigger: schedule a debounced commit while the edit is valid; cancel (defer) while
  // it is invalid or empty so nothing bad is ever persisted, and skip the exact value that just
  // failed so a persistent error doesn't loop. Clear a stale error banner once the user edits
  // away from the failed value.
  useEffect(() => {
    if (commitMode !== 'autosave') return;
    if (!canCommit || currentSignature === failedSignature) {
      debouncedCommit.cancel();
      return;
    }
    if (failedSignature !== null) {
      // The user changed the content after a failure — drop the stale error + block.
      setFailedSignature(null);
      setErrorMessage(undefined);
    }
    const payload = buildPayload();
    if (payload) debouncedCommit(payload);
  }, [commitMode, canCommit, currentSignature, failedSignature, buildPayload, debouncedCommit]);

  // Flush a pending autosave when the record changes or the panel unmounts, so an in-flight
  // debounce isn't dropped on the floor. The debounce replays the last *scheduled* payload, so
  // this commits the outgoing record even though `record` is about to change.
  useEffect(() => () => debouncedCommit.flush(), [record?.dataset_record_id, debouncedCommit]);

  // Stop saying "saved" the moment the user edits again.
  useEffect(() => {
    if (isDirty) setJustSaved(false);
  }, [isDirty]);

  // Clear banners when the drawer switches to a different record.
  useEffect(() => {
    setErrorMessage(undefined);
    setJustSaved(false);
    setFailedSignature(null);
  }, [record?.dataset_record_id]);

  const status: SaveStatus = useMemo(() => {
    if (updateMutation.isLoading || isVerifying) return 'saving';
    if (errorMessage) return 'error';
    if (anyInvalid) return 'invalid';
    if (inputsEmpty && isDirty) return 'empty-inputs';
    // In autosave mode a dirty edit is on its way to the server, so present it as 'saving'
    // rather than 'dirty' (there is no manual save to prompt for). Explicit mode keeps 'dirty'.
    if (isDirty) return commitMode === 'autosave' ? 'saving' : 'dirty';
    if (justSaved) return 'saved';
    return 'clean';
  }, [updateMutation.isLoading, isVerifying, errorMessage, anyInvalid, inputsEmpty, isDirty, justSaved, commitMode]);

  // Commit immediately (Cmd/Ctrl-S, or the explicit-mode Save button).
  const save = useCallback(() => {
    if (!canCommit) return;
    const payload = buildPayload();
    if (!payload) return;
    debouncedCommit.cancel();
    commit(payload);
  }, [canCommit, buildPayload, debouncedCommit, commit]);

  const flush = useCallback(() => debouncedCommit.flush(), [debouncedCommit]);

  const discard = useCallback(() => {
    debouncedCommit.cancel();
    inputs.reset();
    expectations.reset();
    setErrorMessage(undefined);
  }, [debouncedCommit, inputs, expectations]);

  const onContainerKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 's') {
        event.preventDefault();
        save();
      }
    },
    [save],
  );

  return { inputs, expectations, status, errorMessage, isDirty, hasContent, save, flush, discard, onContainerKeyDown };
};
