/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck — punting test typing; see PR2 plan in branch import { renderHook } from '@testing-library/react';
import { createRef } from 'react';
import { useSlashFocusSearch } from './useSlashFocusSearch';
import { describe } from '@jest/globals';
import { beforeEach } from '@jest/globals';
import { afterEach } from '@jest/globals';
import { it } from '@jest/globals';
import { expect } from '@jest/globals';

const fireSlash = (target: EventTarget = document.body, init: KeyboardEventInit = {}) => {
  const event = new KeyboardEvent('keydown', { key: '/', bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  return event;
};

describe('useSlashFocusSearch', () => {
  let input: HTMLInputElement;

  beforeEach(() => {
    input = document.createElement('input');
    document.body.appendChild(input);
  });

  afterEach(() => {
    document.body.removeChild(input);
  });

  it('focuses the input when "/" is pressed on a non-editable element', () => {
    const ref = createRef<HTMLInputElement>();
    Object.defineProperty(ref, 'current', { value: input, writable: true });
    renderHook(() => useSlashFocusSearch(ref));

    expect(document.activeElement).not.toBe(input);
    const event = fireSlash(document.body);
    expect(document.activeElement).toBe(input);
    expect(event.defaultPrevented).toBe(true);
  });

  it('does not steal focus when the user is typing into an input', () => {
    const ref = createRef<HTMLInputElement>();
    Object.defineProperty(ref, 'current', { value: input, writable: true });
    renderHook(() => useSlashFocusSearch(ref));

    const otherInput = document.createElement('input');
    document.body.appendChild(otherInput);
    otherInput.focus();

    const event = fireSlash(otherInput);
    expect(document.activeElement).toBe(otherInput);
    expect(event.defaultPrevented).toBe(false);
    document.body.removeChild(otherInput);
  });

  it('does not steal focus when the user is typing into a textarea', () => {
    const ref = createRef<HTMLInputElement>();
    Object.defineProperty(ref, 'current', { value: input, writable: true });
    renderHook(() => useSlashFocusSearch(ref));

    const textarea = document.createElement('textarea');
    document.body.appendChild(textarea);
    textarea.focus();

    const event = fireSlash(textarea);
    expect(document.activeElement).toBe(textarea);
    expect(event.defaultPrevented).toBe(false);
    document.body.removeChild(textarea);
  });

  it('ignores "/" when a modifier key is held (Ctrl+/ / Cmd+/ stay free)', () => {
    const ref = createRef<HTMLInputElement>();
    Object.defineProperty(ref, 'current', { value: input, writable: true });
    renderHook(() => useSlashFocusSearch(ref));

    const event = fireSlash(document.body, { ctrlKey: true });
    expect(document.activeElement).not.toBe(input);
    expect(event.defaultPrevented).toBe(false);

    const event2 = fireSlash(document.body, { metaKey: true });
    expect(document.activeElement).not.toBe(input);
    expect(event2.defaultPrevented).toBe(false);
  });

  it('is a no-op when the ref is empty', () => {
    const ref = createRef<HTMLInputElement>();
    renderHook(() => useSlashFocusSearch(ref));

    const event = fireSlash(document.body);
    expect(document.activeElement).not.toBe(input);
    // We still preventDefault so the slash doesn't quick-find in the page, mirroring the
    // user-facing behavior when the input is just briefly unmounted.
    expect(event.defaultPrevented).toBe(true);
  });

  it('does not steal focus from inside an open Drawer/Modal (keeps the focus trap intact)', () => {
    const ref = createRef<HTMLInputElement>();
    Object.defineProperty(ref, 'current', { value: input, writable: true });
    renderHook(() => useSlashFocusSearch(ref));

    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('data-state', 'open');
    const insideDialog = document.createElement('div');
    dialog.appendChild(insideDialog);
    document.body.appendChild(dialog);

    const event = fireSlash(insideDialog);
    expect(document.activeElement).not.toBe(input);
    expect(event.defaultPrevented).toBe(false);
    document.body.removeChild(dialog);
  });

  it('removes the listener on unmount', () => {
    const ref = createRef<HTMLInputElement>();
    Object.defineProperty(ref, 'current', { value: input, writable: true });
    const { unmount } = renderHook(() => useSlashFocusSearch(ref));

    unmount();
    const event = fireSlash(document.body);
    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).not.toBe(input);
  });
});
