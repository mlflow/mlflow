import { describe, expect, test, beforeEach } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useTraceColumnVisibility } from './useTraceColumnVisibility';
import type { TraceColumnId } from '../types';

// Default set for the test: everything visible except trace_id/tokens/cost, and session only when
// the page has sessions (a data-driven default, passed in by the consumer).
const makeGetDefaultVisible =
  (hasSessionOnPage: boolean) =>
  (id: TraceColumnId): boolean => {
    if (id === 'session') {
      return hasSessionOnPage;
    }
    return id !== 'trace_id' && id !== 'tokens' && id !== 'cost';
  };

const KEY = 'test.traces-table.columns';

describe('useTraceColumnVisibility', () => {
  beforeEach(() => window.localStorage.clear());

  test('uses the computed defaults when nothing is stored', () => {
    const { result } = renderHook(() =>
      useTraceColumnVisibility({ storageKey: KEY, version: 1, getDefaultVisible: makeGetDefaultVisible(false) }),
    );
    expect(result.current.visibleColumns).toEqual([
      'trace_name',
      'start_time',
      'input',
      'output',
      'user',
      'duration',
      'state',
      'source',
      'run_name',
      'tags',
    ]);
  });

  test('session default follows the data-driven flag', () => {
    const { result } = renderHook(() =>
      useTraceColumnVisibility({ storageKey: KEY, version: 1, getDefaultVisible: makeGetDefaultVisible(true) }),
    );
    expect(result.current.visibleColumns).toContain('session');
  });

  test('toggling a column writes a sticky override that wins over the default', () => {
    const { result } = renderHook(() =>
      useTraceColumnVisibility({ storageKey: KEY, version: 1, getDefaultVisible: makeGetDefaultVisible(false) }),
    );
    // trace_id is hidden by default; toggling shows it.
    act(() => result.current.toggleColumn('trace_id'));
    expect(result.current.visibleColumns).toContain('trace_id');
    // The override persists in canonical order (trace_id is first).
    expect(result.current.visibleColumns[0]).toBe('trace_id');
  });

  test('reset clears overrides, returning to the defaults', () => {
    const { result } = renderHook(() =>
      useTraceColumnVisibility({ storageKey: KEY, version: 1, getDefaultVisible: makeGetDefaultVisible(false) }),
    );
    act(() => result.current.toggleColumn('input')); // hide input
    expect(result.current.visibleColumns).not.toContain('input');
    act(() => result.current.resetToDefaults());
    expect(result.current.visibleColumns).toContain('input');
  });

  test('setColumns adopts an explicit visible set, independent of the live default', () => {
    // Start with sessions in the default (session would default visible), then adopt a set that
    // omits it: session must end hidden (pinned false), not fall back to its default.
    const { result } = renderHook(() =>
      useTraceColumnVisibility({ storageKey: KEY, version: 1, getDefaultVisible: makeGetDefaultVisible(true) }),
    );
    act(() => result.current.setColumns(['trace_id', 'input']));
    expect(result.current.visibleColumns).toEqual(['trace_id', 'input']);
    expect(result.current.visibleColumns).not.toContain('session');
    // A subsequent reset returns to the live defaults (session back, per the flag).
    act(() => result.current.resetToDefaults());
    expect(result.current.visibleColumns).toContain('session');
  });

  test('columns render in canonical order regardless of toggle order', () => {
    const { result } = renderHook(() =>
      useTraceColumnVisibility({ storageKey: KEY, version: 1, getDefaultVisible: makeGetDefaultVisible(false) }),
    );
    act(() => result.current.toggleColumn('cost')); // show cost (canonically after tokens, before tags)
    act(() => result.current.toggleColumn('trace_id')); // show trace_id (canonically first)
    const visible = result.current.visibleColumns;
    // trace_id precedes cost precedes tags in the canonical order.
    expect(visible.indexOf('trace_id')).toBeLessThan(visible.indexOf('cost'));
    expect(visible.indexOf('cost')).toBeLessThan(visible.indexOf('tags'));
  });
});
