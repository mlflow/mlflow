import { useCallback, useEffect, useState } from 'react';
import { useDebouncedCallback } from 'use-debounce';

interface UseDebouncedSearchInputParams {
  /** Source-of-truth search value from the URL or upstream state. */
  committedValue: string;
  /** Called when the debounced write should commit (writes back to URL state). */
  onCommit: (next: string) => void;
  debounceMs: number;
}

export interface UseDebouncedSearchInputResult {
  /** Current input value — drives the controlled `<Input value={…}>`. */
  input: string;
  /** Update the input and schedule a debounced commit. */
  setInput: (next: string) => void;
  /** Cancel any pending commit and immediately commit the empty string. */
  clear: () => void;
  /** Execute the pending debounced commit synchronously. */
  flush: () => void;
}

/**
 * Tracks the local input value for a debounced search box and exposes `flush()` so callers
 * (e.g. pagination clicks, page-clamp effects) can synchronously commit a pending write
 * before triggering a sibling URL transition. Without flushing, a pending search debounce
 * can land *after* a page-index write and clobber it (search resets `?page` back to 1).
 */
export const useDebouncedSearchInput = ({
  committedValue,
  onCommit,
  debounceMs,
}: UseDebouncedSearchInputParams): UseDebouncedSearchInputResult => {
  const [input, setInputState] = useState(committedValue);

  // Resync local state with upstream (e.g. browser back/forward changes the URL).
  useEffect(() => {
    setInputState(committedValue);
  }, [committedValue]);

  const debouncedCommit = useDebouncedCallback(onCommit, debounceMs);

  // Drop any pending write on unmount so it can't pollute the destination page after navigation.
  useEffect(() => () => debouncedCommit.cancel(), [debouncedCommit]);

  const setInput = useCallback(
    (next: string) => {
      setInputState(next);
      debouncedCommit(next);
    },
    [debouncedCommit],
  );

  const clear = useCallback(() => {
    debouncedCommit.cancel();
    setInputState('');
    onCommit('');
  }, [debouncedCommit, onCommit]);

  const flush = useCallback(() => {
    debouncedCommit.flush();
  }, [debouncedCommit]);

  return { input, setInput, clear, flush };
};
