import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useDebouncedSearchInput } from './useDebouncedSearchInput';

describe('useDebouncedSearchInput', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('input mirrors the committed value on first render', () => {
    const { result } = renderHook(() =>
      useDebouncedSearchInput({ committedValue: 'hello', onCommit: jest.fn(), debounceMs: 100 }),
    );
    expect(result.current.input).toBe('hello');
  });

  test('setInput updates the local value immediately and commits after the debounce window', () => {
    const onCommit = jest.fn();
    const { result } = renderHook(() => useDebouncedSearchInput({ committedValue: '', onCommit, debounceMs: 100 }));

    act(() => result.current.setInput('a'));
    expect(result.current.input).toBe('a');
    expect(onCommit).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(99);
    });
    expect(onCommit).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(1);
    });
    expect(onCommit).toHaveBeenCalledWith('a');
  });

  test('flush executes the pending commit synchronously', () => {
    const onCommit = jest.fn();
    const { result } = renderHook(() => useDebouncedSearchInput({ committedValue: '', onCommit, debounceMs: 250 }));

    act(() => result.current.setInput('hello'));
    expect(onCommit).not.toHaveBeenCalled();

    act(() => result.current.flush());
    expect(onCommit).toHaveBeenCalledWith('hello');
  });

  test('clear cancels the pending commit and immediately commits the empty string', () => {
    const onCommit = jest.fn();
    const { result } = renderHook(() => useDebouncedSearchInput({ committedValue: '', onCommit, debounceMs: 100 }));

    act(() => result.current.setInput('typing'));
    act(() => result.current.clear());

    expect(result.current.input).toBe('');
    expect(onCommit).toHaveBeenCalledWith('');
    // Advance past the original debounce window — the cancelled write must NOT fire.
    act(() => {
      jest.advanceTimersByTime(200);
    });
    expect(onCommit).toHaveBeenCalledTimes(1);
  });

  test('upstream committedValue changes resync the local input (e.g. browser back)', () => {
    const onCommit = jest.fn();
    const { result, rerender } = renderHook(
      ({ committed }: { committed: string }) =>
        useDebouncedSearchInput({ committedValue: committed, onCommit, debounceMs: 100 }),
      { initialProps: { committed: '' } },
    );
    act(() => result.current.setInput('typing'));
    expect(result.current.input).toBe('typing');

    rerender({ committed: 'from-url' });
    expect(result.current.input).toBe('from-url');
  });

  test('unmount cancels any pending commit', () => {
    const onCommit = jest.fn();
    const { result, unmount } = renderHook(() =>
      useDebouncedSearchInput({ committedValue: '', onCommit, debounceMs: 100 }),
    );
    act(() => result.current.setInput('typing'));
    unmount();
    act(() => {
      jest.advanceTimersByTime(200);
    });
    expect(onCommit).not.toHaveBeenCalled();
  });
});
