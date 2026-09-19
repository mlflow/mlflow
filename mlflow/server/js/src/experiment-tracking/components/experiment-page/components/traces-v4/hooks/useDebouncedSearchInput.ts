import { useCallback, useEffect, useRef, useState } from 'react';
import { useDebouncedCallback } from 'use-debounce';

interface UseDebouncedSearchInputParams {
  /** Source-of-truth search value from the URL or upstream state. */
  committedValue: string;
  /** Called when the debounced write should commit (writes back to URL state). */
  onCommit: (next: string) => void;
  debounceMs: number;
  /**
   * Whether `setInput` schedules a debounced commit. Defaults to `true` (commit-as-you-type). Pass
   * `false` for a commit-on-submit box: `setInput` only updates local state, and `submit`/`clear`
   * drive the actual commit.
   */
  commitOnChange?: boolean;
}

export interface UseDebouncedSearchInputResult {
  /** Current input value — drives the controlled `<Input value={…}>`. */
  input: string;
  /** Update the input and (when `commitOnChange`) schedule a debounced commit. */
  setInput: (next: string) => void;
  /** Cancel any pending commit and immediately commit the empty string. */
  clear: () => void;
  /** Execute the pending debounced commit synchronously. */
  flush: () => void;
  /** Cancel any pending commit and immediately commit the latest input (mirrors `clear` for Enter). */
  submit: () => void;
}

/**
 * V4-local search-input hook. The OSS datasets-v2 `useDebouncedSearchInput` is commit-as-you-type
 * only; the V4 traces search box commits on Enter (submit) and on clear, so this adds the
 * `commitOnChange` toggle + `submit`. Kept inside the V4 dir rather than editing the shared
 * datasets-v2 hook (a different consumer).
 */
export const useDebouncedSearchInput = ({
  committedValue,
  onCommit,
  debounceMs,
  commitOnChange = true,
}: UseDebouncedSearchInputParams): UseDebouncedSearchInputResult => {
  const [input, setInputState] = useState(committedValue);

  // Latest input, kept in a ref so `submit` reads the current value without depending on `input`.
  const inputRef = useRef(input);

  // Resync local state with upstream (e.g. browser back/forward changes the URL).
  useEffect(() => {
    setInputState(committedValue);
    inputRef.current = committedValue;
  }, [committedValue]);

  const debouncedCommit = useDebouncedCallback(onCommit, debounceMs);

  // Drop any pending write on unmount so it can't pollute the destination page after navigation.
  useEffect(() => () => debouncedCommit.cancel(), [debouncedCommit]);

  const setInput = useCallback(
    (next: string) => {
      setInputState(next);
      inputRef.current = next;
      if (commitOnChange) {
        debouncedCommit(next);
      }
    },
    [debouncedCommit, commitOnChange],
  );

  const clear = useCallback(() => {
    debouncedCommit.cancel();
    setInputState('');
    inputRef.current = '';
    onCommit('');
  }, [debouncedCommit, onCommit]);

  const flush = useCallback(() => {
    debouncedCommit.flush();
  }, [debouncedCommit]);

  const submit = useCallback(() => {
    debouncedCommit.cancel();
    onCommit(inputRef.current);
  }, [debouncedCommit, onCommit]);

  return { input, setInput, clear, flush, submit };
};
