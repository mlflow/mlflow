import { useEffect, type RefObject } from 'react';

interface FocusableRef {
  focus: () => void;
}

/**
 * Wires a document-level "/" hotkey that focuses the search input — matches the convention
 * users expect from the Traces tab and other dense list views. Skips when the user is
 * already typing into an editable element and when modifiers are held, so it doesn't
 * trample search-and-replace, save-page, or in-input slashes.
 *
 * The ref's target only needs a `.focus()` method, so callers can point this at any of
 * `HTMLInputElement`, Dubois's `InputRef` (which forwards to the antd input instance), or
 * a custom focusable wrapper.
 */
export const useSlashFocusSearch = (inputRef: RefObject<FocusableRef | null>): void => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== '/' || event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      const target = event.target as HTMLElement | null;
      const isEditable =
        target !== null &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);
      if (isEditable) {
        return;
      }
      // Don't pull focus out of an open Drawer/Modal — it would break the focus trap and
      // strand the user outside the dialog they're trying to interact with.
      if (target?.closest('[role="dialog"]')) {
        return;
      }
      event.preventDefault();
      inputRef.current?.focus();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [inputRef]);
};
