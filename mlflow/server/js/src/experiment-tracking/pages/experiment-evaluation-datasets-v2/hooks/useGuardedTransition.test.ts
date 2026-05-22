// @ts-nocheck — punting test typing; see PR2 plan in branch import { describe, expect, jest, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useGuardedTransition } from './useGuardedTransition';

describe('useGuardedTransition', () => {
  test('clean request fires the transition immediately and does not open the prompt', () => {
    const transition = jest.fn();
    const { result } = renderHook(() => useGuardedTransition({ isDirty: false }));

    act(() => result.current.requestTransition(transition));

    expect(transition).toHaveBeenCalledTimes(1);
    expect(result.current.isPromptOpen).toBe(false);
  });

  test('dirty request opens the prompt and defers the transition', () => {
    const transition = jest.fn();
    const { result } = renderHook(() => useGuardedTransition({ isDirty: true }));

    act(() => result.current.requestTransition(transition));

    expect(transition).not.toHaveBeenCalled();
    expect(result.current.isPromptOpen).toBe(true);
  });

  test('confirm runs the stashed transition and closes the prompt', () => {
    const transition = jest.fn();
    const { result } = renderHook(() => useGuardedTransition({ isDirty: true }));

    act(() => result.current.requestTransition(transition));
    act(() => result.current.confirm());

    expect(transition).toHaveBeenCalledTimes(1);
    expect(result.current.isPromptOpen).toBe(false);
  });

  test('cancel closes the prompt without firing the stashed transition', () => {
    const transition = jest.fn();
    const { result } = renderHook(() => useGuardedTransition({ isDirty: true }));

    act(() => result.current.requestTransition(transition));
    act(() => result.current.cancel());

    expect(transition).not.toHaveBeenCalled();
    expect(result.current.isPromptOpen).toBe(false);
  });

  test('confirm with no pending transition is a safe no-op', () => {
    const { result } = renderHook(() => useGuardedTransition({ isDirty: true }));

    expect(() => act(() => result.current.confirm())).not.toThrow();
    expect(result.current.isPromptOpen).toBe(false);
  });

  test('a second dirty request before confirm replaces the stashed transition', () => {
    const first = jest.fn();
    const second = jest.fn();
    const { result } = renderHook(() => useGuardedTransition({ isDirty: true }));

    act(() => result.current.requestTransition(first));
    act(() => result.current.requestTransition(second));
    act(() => result.current.confirm());

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });
});