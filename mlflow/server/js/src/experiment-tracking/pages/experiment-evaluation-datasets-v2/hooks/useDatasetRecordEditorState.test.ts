// @ts-nocheck — punting test typing; see PR2 plan in branch import { describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useDatasetRecordEditorState } from './useDatasetRecordEditorState';

describe('useDatasetRecordEditorState', () => {
  test('initial state is clean and matches the stringified initial value', () => {
    const { result } = renderHook(() => useDatasetRecordEditorState({ recordId: 'r1', initialValue: { foo: 'bar' } }));
    expect(result.current.isDirty).toBe(false);
    expect(result.current.isValid).toBe(true);
    expect(result.current.text).toBe(JSON.stringify({ foo: 'bar' }, null, 2));
    expect(result.current.parsed).toEqual({ foo: 'bar' });
  });

  test('editing the text marks the state dirty', () => {
    const { result } = renderHook(() => useDatasetRecordEditorState({ recordId: 'r1', initialValue: { foo: 'bar' } }));
    act(() => result.current.setText('{"foo":"baz"}'));
    expect(result.current.isDirty).toBe(true);
    expect(result.current.parsed).toEqual({ foo: 'baz' });
  });

  test('invalid JSON marks the state invalid and clears parsed', () => {
    const { result } = renderHook(() => useDatasetRecordEditorState({ recordId: 'r1', initialValue: { foo: 'bar' } }));
    act(() => result.current.setText('{not json'));
    expect(result.current.isValid).toBe(false);
    expect(result.current.parsed).toBeUndefined();
    // parseError surfaces the underlying JSON.parse message so callers can show diagnostics.
    expect(result.current.parseError).toBeDefined();
    expect(typeof result.current.parseError).toBe('string');
  });

  test('arrays and primitives are treated as invalid for record fields', () => {
    const { result } = renderHook(() => useDatasetRecordEditorState({ recordId: 'r1', initialValue: { foo: 'bar' } }));
    act(() => result.current.setText('[1, 2, 3]'));
    expect(result.current.isValid).toBe(false);
    expect(result.current.parseError).toMatch(/object/i);
    act(() => result.current.setText('"a string"'));
    expect(result.current.isValid).toBe(false);
    expect(result.current.parseError).toMatch(/object/i);
  });

  test('parseError is undefined when text is valid JSON or empty', () => {
    const { result } = renderHook(() => useDatasetRecordEditorState({ recordId: 'r1', initialValue: { foo: 'bar' } }));
    expect(result.current.parseError).toBeUndefined();
    act(() => result.current.setText(''));
    expect(result.current.parseError).toBeUndefined();
    act(() => result.current.setText('{"a": 1}'));
    expect(result.current.parseError).toBeUndefined();
  });

  test('empty text parses to the empty object so saves commit an explicit value', () => {
    const { result } = renderHook(() => useDatasetRecordEditorState({ recordId: 'r1', initialValue: { foo: 'bar' } }));
    act(() => result.current.setText(''));
    expect(result.current.isValid).toBe(true);
    expect(result.current.parsed).toEqual({});
    expect(result.current.isDirty).toBe(true);
  });

  test('reset returns the editor to the initial text and clears dirty', () => {
    const { result } = renderHook(() => useDatasetRecordEditorState({ recordId: 'r1', initialValue: { foo: 'bar' } }));
    act(() => result.current.setText('{"foo":"baz"}'));
    expect(result.current.isDirty).toBe(true);
    act(() => result.current.reset());
    expect(result.current.isDirty).toBe(false);
    expect(result.current.parsed).toEqual({ foo: 'bar' });
  });

  test('changing the record id resets local state', () => {
    const { result, rerender } = renderHook(
      ({ id, value }: { id: string; value: Record<string, unknown> }) =>
        useDatasetRecordEditorState({ recordId: id, initialValue: value }),
      { initialProps: { id: 'r1', value: { foo: 'bar' } as Record<string, unknown> } },
    );
    act(() => result.current.setText('{"foo":"baz"}'));
    expect(result.current.isDirty).toBe(true);

    rerender({ id: 'r2', value: { hello: 'world' } as Record<string, unknown> });
    expect(result.current.isDirty).toBe(false);
    expect(result.current.parsed).toEqual({ hello: 'world' });
  });

  test('refetching the same record (different reference, same id) preserves in-flight edits', () => {
    const { result, rerender } = renderHook(
      ({ id, value }: { id: string; value: Record<string, unknown> }) =>
        useDatasetRecordEditorState({ recordId: id, initialValue: value }),
      { initialProps: { id: 'r1', value: { foo: 'bar' } as Record<string, unknown> } },
    );
    act(() => result.current.setText('{"foo":"baz"}'));
    expect(result.current.isDirty).toBe(true);

    // Simulate a refetch returning a new object reference for the same record.
    rerender({ id: 'r1', value: { foo: 'bar' } as Record<string, unknown> });
    expect(result.current.text).toBe('{"foo":"baz"}');
    expect(result.current.isDirty).toBe(true);
  });

  test('server-side content change without local edits stays clean (no false-dirty)', () => {
    const { result, rerender } = renderHook(
      ({ id, value }: { id: string; value: Record<string, unknown> }) =>
        useDatasetRecordEditorState({ recordId: id, initialValue: value }),
      { initialProps: { id: 'r1', value: { a: 1 } as Record<string, unknown> } },
    );
    expect(result.current.isDirty).toBe(false);

    // Concurrent edit lands in the cache: same record id, different content.
    // The user has not typed — `isDirty` must stay false so the Save button doesn't arm
    // against an untouched draft (which would clobber the new server state on click).
    rerender({ id: 'r1', value: { a: 2 } as Record<string, unknown> });
    expect(result.current.isDirty).toBe(false);
    expect(result.current.text).toBe(JSON.stringify({ a: 1 }, null, 2));
  });

  test('reset() after a server-side content change adopts the latest server value', () => {
    const { result, rerender } = renderHook(
      ({ id, value }: { id: string; value: Record<string, unknown> }) =>
        useDatasetRecordEditorState({ recordId: id, initialValue: value }),
      { initialProps: { id: 'r1', value: { a: 1 } as Record<string, unknown> } },
    );
    rerender({ id: 'r1', value: { a: 2 } as Record<string, unknown> });
    act(() => result.current.reset());
    expect(result.current.isDirty).toBe(false);
    expect(result.current.parsed).toEqual({ a: 2 });
  });

  test('reset(nextBaseline) advances to the explicit value and ignores latestInitialTextRef', () => {
    const { result, rerender } = renderHook(
      ({ id, value }: { id: string; value: Record<string, unknown> }) =>
        useDatasetRecordEditorState({ recordId: id, initialValue: value }),
      { initialProps: { id: 'r1', value: { a: 1 } as Record<string, unknown> } },
    );
    // Concurrent server state changes the ref, but the explicit baseline wins.
    rerender({ id: 'r1', value: { a: 99 } as Record<string, unknown> });
    act(() => result.current.setText('{"a":42}'));
    act(() => result.current.reset('{"a":42}'));
    expect(result.current.isDirty).toBe(false);
    expect(result.current.text).toBe('{"a":42}');
    expect(result.current.parsed).toEqual({ a: 42 });
  });
});