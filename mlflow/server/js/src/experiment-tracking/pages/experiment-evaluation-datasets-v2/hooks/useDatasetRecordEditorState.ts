import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

interface UseDatasetRecordEditorStateParams {
  /**
   * Stable identifier for the source record. Editor state resets when this changes.
   * Pass `undefined` for "new record" flows — the editor then sticks to its initial value
   * for the lifetime of the mount.
   */
  recordId?: string;
  /** Source-of-truth dictionary. Read once per `recordId` change, ignored on refetch. */
  initialValue: Record<string, unknown> | undefined;
}

interface UseDatasetRecordEditorStateResult {
  /** JSON string the editor displays (and mutates via onChange). */
  text: string;
  setText: (next: string) => void;
  /** Parsed value when the text is valid JSON. Empty text parses to `{}`. */
  parsed: Record<string, unknown> | undefined;
  isDirty: boolean;
  isValid: boolean;
  /**
   * Raw parse error message (e.g. `"Unexpected token } in JSON at position 47"`) when the
   * text is invalid, otherwise `undefined`. Surfaced for diagnostics — UI surfaces typically
   * show a localized "Invalid JSON" string driven by `!isValid` instead.
   */
  parseError: string | undefined;
  /**
   * Reset both `text` and the dirty `baseline`. Pass `nextBaseline` to advance to an explicit
   * post-save value rather than re-reading `latestInitialTextRef` — needed for save-success
   * paths so the new baseline matches the text the user just submitted, independent of when
   * the upstream re-render flushes.
   */
  reset: (nextBaseline?: string) => void;
}

interface ParseResult {
  value: Record<string, unknown> | undefined;
  error: string | undefined;
}

const stringifyOrEmpty = (value: Record<string, unknown> | undefined): string =>
  value === undefined ? '' : JSON.stringify(value, null, 2);

const parseRecordObject = (text: string): ParseResult => {
  if (text.trim() === '') return { value: {}, error: undefined };
  try {
    const value = JSON.parse(text);
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      return { value: value as Record<string, unknown>, error: undefined };
    }
    return { value: undefined, error: 'JSON must be an object (not an array or primitive)' };
  } catch (e) {
    return { value: undefined, error: e instanceof Error ? e.message : 'Invalid JSON' };
  }
};

/**
 * Tracks local editor text for one JSON field (inputs, expectations). Surfaces parsed value when
 * valid JSON, dirty flag for save-button enablement, and `reset` for the drawer's Discard button.
 *
 * The dirty comparison uses a `baseline` snapshot that only advances on `recordId` change or
 * explicit `reset()` — never on server-side refetch. That way a concurrent edit landing in the
 * cache cannot silently arm the Save button against an untouched draft and overwrite the new
 * server state on the next click.
 *
 * Empty text is treated as the empty object `{}` so emptying the editor commits an explicit
 * write rather than silently dropping the field.
 */
export const useDatasetRecordEditorState = ({
  recordId,
  initialValue,
}: UseDatasetRecordEditorStateParams): UseDatasetRecordEditorStateResult => {
  const initialTextForCurrentRecord = useMemo(() => stringifyOrEmpty(initialValue), [initialValue]);

  // Track the freshest initialValue without making it a reset/dirty trigger.
  const latestInitialTextRef = useRef(initialTextForCurrentRecord);
  useEffect(() => {
    latestInitialTextRef.current = initialTextForCurrentRecord;
  }, [initialTextForCurrentRecord]);

  const [text, setText] = useState(initialTextForCurrentRecord);
  const [baseline, setBaseline] = useState(initialTextForCurrentRecord);

  // Reset only when the underlying record changes — not on every refetch.
  useEffect(() => {
    setText(latestInitialTextRef.current);
    setBaseline(latestInitialTextRef.current);
  }, [recordId]);

  const { value: parsed, error: parseError } = useMemo(() => parseRecordObject(text), [text]);
  const isValid = parsed !== undefined;
  const isDirty = text !== baseline;

  const reset = useCallback((nextBaseline?: string) => {
    const next = nextBaseline ?? latestInitialTextRef.current;
    setText(next);
    setBaseline(next);
  }, []);

  return { text, setText, parsed, isDirty, isValid, parseError, reset };
};
