import { useCallback, useEffect, useRef, useState } from 'react';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';

export const AUTOSAVE_DELAY_MS = 500;

export type SettingSaveError = { kind: 'invalid'; message: string } | { kind: 'request'; reason?: string };

export interface UseSettingAutosaveOptions<T> {
  /** The value the server has. */
  savedValue: T;
  isEditable: boolean;
  /** Values with the same key are the same setting value, so only one of them is saved. */
  getKey: (value: T) => string;
  /** An error message for a value that can't be saved. */
  validate?: (value: T) => string | undefined;
  save: (value: T) => Promise<void>;
}

/**
 * Saves a setting's draft shortly after each edit, or right away with `saveNow` (for example on blur).
 * Saves run one at a time in order, and a value that is already queued or saved isn't sent again.
 */
export const useSettingAutosave = <T>({
  savedValue,
  isEditable,
  getKey,
  validate,
  save,
}: UseSettingAutosaveOptions<T>) => {
  const [draft, setDraft] = useState(savedValue);
  const [error, setError] = useState<SettingSaveError | null>(null);
  // Read through a ref so new callbacks from a parent re-render don't restart the autosave timer.
  const optionsRef = useRef({ savedValue, getKey, validate, save });
  optionsRef.current = { savedValue, getKey, validate, save };
  const savedKey = getKey(savedValue);
  const draftKey = getKey(draft);
  const persistedKeyRef = useRef(savedKey);
  const latestDraftRef = useRef(savedValue);
  // The value the server will have once every queued save has run.
  const latestQueuedKeyRef = useRef(savedKey);
  const pendingSaveCountRef = useRef(0);
  const saveQueueRef = useRef(Promise.resolve());
  const autosaveTimerRef = useRef<ReturnType<typeof setTimeout>>();

  const queueSave = useCallback(
    (value: T) => {
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
      }
      const key = optionsRef.current.getKey(value);
      // Compare with the queued value, not the saved one: restoring the saved value while another
      // save is in flight still has to be sent, or that other save wins.
      if (!isEditable || key === latestQueuedKeyRef.current) {
        return;
      }
      const invalidMessage = optionsRef.current.validate?.(value);
      if (invalidMessage) {
        setError({ kind: 'invalid', message: invalidMessage });
        return;
      }

      latestQueuedKeyRef.current = key;
      setError(null);
      pendingSaveCountRef.current += 1;
      const isLatestDraft = () => optionsRef.current.getKey(latestDraftRef.current) === key;
      const pendingSave = saveQueueRef.current.then(async () => {
        try {
          if (key !== latestQueuedKeyRef.current || key === persistedKeyRef.current) {
            return;
          }
          await optionsRef.current.save(value);
          persistedKeyRef.current = key;
          if (isLatestDraft()) {
            setError(null);
          }
        } catch (saveError) {
          if (isLatestDraft()) {
            setError({
              kind: 'request',
              reason:
                saveError instanceof ErrorWrapper
                  ? saveError.getMessageField()
                  : saveError instanceof Error
                    ? saveError.message
                    : undefined,
            });
          }
        } finally {
          pendingSaveCountRef.current -= 1;
        }
      });
      saveQueueRef.current = pendingSave.catch(() => undefined);
    },
    [isEditable],
  );

  useEffect(() => {
    latestDraftRef.current = draft;
    if (!isEditable || draftKey === latestQueuedKeyRef.current) {
      return;
    }
    autosaveTimerRef.current = setTimeout(() => queueSave(draft), AUTOSAVE_DELAY_MS);
    return () => {
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
      }
    };
  }, [draft, draftKey, isEditable, queueSave]);

  useEffect(() => {
    // A value that arrives while saves are pending is the refetch after one of them; adopting it
    // could overwrite a newer draft that is still queued.
    if (pendingSaveCountRef.current > 0) {
      return;
    }
    if (optionsRef.current.getKey(latestDraftRef.current) === persistedKeyRef.current) {
      setDraft(optionsRef.current.savedValue);
      latestDraftRef.current = optionsRef.current.savedValue;
    }
    persistedKeyRef.current = savedKey;
    latestQueuedKeyRef.current = savedKey;
  }, [savedKey]);

  const updateDraft = useCallback((value: T) => {
    setError(null);
    setDraft(value);
  }, []);

  /** Saves `value`, or the current draft, without waiting for the autosave delay. */
  const saveNow = useCallback((value?: T) => queueSave(value ?? latestDraftRef.current), [queueSave]);

  const retry = useCallback(() => {
    latestQueuedKeyRef.current = persistedKeyRef.current;
    queueSave(latestDraftRef.current);
  }, [queueSave]);

  return { draft, updateDraft, saveNow, retry, error };
};
