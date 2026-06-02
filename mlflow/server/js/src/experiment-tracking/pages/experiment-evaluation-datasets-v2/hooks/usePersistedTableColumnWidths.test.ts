import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import type { ColumnSizingState } from '@tanstack/react-table';
import { usePersistedTableColumnWidths } from './usePersistedTableColumnWidths';

// Mirror `useLocalStorage`'s key format: `${key}_v${version}`. Knowing the
// shape lets us pre-seed corrupt entries directly (no envelope wrapping —
// `useLocalStorage` writes raw `JSON.stringify(value)`), and read back the
// raw persisted value to verify what landed in storage after a setter call.
const STORAGE_VERSION = 1;
const buildStorageKey = (experimentId: string, datasetId: string) =>
  `mlflow.eval-datasets.column-widths.${experimentId}.${datasetId}_v${STORAGE_VERSION}`;

const seedRawWidths = (experimentId: string, datasetId: string, raw: Record<string, unknown>) => {
  window.localStorage.setItem(buildStorageKey(experimentId, datasetId), JSON.stringify(raw));
};

const readRawWidths = (experimentId: string, datasetId: string): unknown => {
  const raw = window.localStorage.getItem(buildStorageKey(experimentId, datasetId));
  return raw === null ? null : JSON.parse(raw);
};

describe('usePersistedTableColumnWidths', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  test('starts empty when nothing is persisted', () => {
    // Defaults live on the ColumnDefs, so the stored map begins empty and only dragged
    // columns ever get written.
    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));
    expect(result.current.columnSizing).toEqual({});
  });

  test('setColumnSizing stores a direct value', () => {
    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));

    act(() => {
      result.current.setColumnSizing({ inputs: 512 });
    });

    expect(result.current.columnSizing).toEqual({ inputs: 512 });
  });

  test('setColumnSizing accepts a functional updater (the shape TanStack calls it with)', () => {
    // onColumnSizingChange hands the setter a `(prev) => next` updater on every drag tick;
    // this guards that the useLocalStorage setter plugs in directly with no adapter.
    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));

    act(() => {
      result.current.setColumnSizing({ inputs: 512 });
    });
    act(() => {
      result.current.setColumnSizing((prev) => ({ ...prev, expectations: 320 }));
    });

    expect(result.current.columnSizing).toEqual({ inputs: 512, expectations: 320 });
  });

  test('persists across remounts for the same dataset', () => {
    const params = { experimentId: 'exp-1', datasetId: 'ds-1' };

    const first = renderHook(() => usePersistedTableColumnWidths(params));
    act(() => {
      first.result.current.setColumnSizing({ inputs: 256 });
    });
    first.unmount();

    const second = renderHook(() => usePersistedTableColumnWidths(params));
    expect(second.result.current.columnSizing).toEqual({ inputs: 256 });
  });

  test('widths are isolated per dataset', () => {
    const first = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));
    act(() => {
      first.result.current.setColumnSizing({ inputs: 256 });
    });

    // Different dataset → its own key → starts empty.
    const second = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-2' }));
    expect(second.result.current.columnSizing).toEqual({});
  });

  test('widths are isolated per experiment', () => {
    const first = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));
    act(() => {
      first.result.current.setColumnSizing({ inputs: 256 });
    });

    const second = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-2', datasetId: 'ds-1' }));
    expect(second.result.current.columnSizing).toEqual({});
  });

  test('filters corrupt entries pre-seeded in localStorage on read', () => {
    // Schema-less storage means corrupt entries (negatives, strings, zero, null)
    // can land in localStorage — from older versions, hand-written debug data, or
    // a buggy seed (the original `useLayoutEffect` would persist `0` widths if it
    // ran inside a `display: none` ancestor). Guard that the sanitizer strips them
    // before they flow into `flex: 0 0 ${px}px` and collapse a column. NaN can't
    // round-trip through JSON, so it isn't covered here — the in-memory test
    // path below exercises the equivalent code branch.
    seedRawWidths('exp-1', 'ds-1', {
      inputs: 400, //         valid → keep
      source: -10, //         invalid: negative
      tags: 'foo', //         invalid: not a number
      create_time: 0, //      invalid: zero (would clamp up to minSize and lie about the user's intent)
      last_updated: null, //  invalid: null
    });

    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));
    expect(result.current.columnSizing).toEqual({ inputs: 400 });
  });

  test('does not persist corrupt entries when the setter is called with a direct value', () => {
    // Defends against a caller writing through bad data (TypeScript can be cast away,
    // and some upstream flows compute widths that could become NaN/negative). The
    // wrapper sanitises the value before handing it to useLocalStorage.
    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));

    act(() => {
      result.current.setColumnSizing({
        inputs: 400,
        source: -10,
        tags: 'foo' as unknown as number,
      } as ColumnSizingState);
    });

    // Read the raw persisted value rather than `result.current.columnSizing`: the
    // read path also sanitises, so checking it doesn't prove the write path did.
    expect(readRawWidths('exp-1', 'ds-1')).toEqual({ inputs: 400 });
  });

  test('strips corrupt prev entries when TanStack invokes the functional updater', () => {
    // Pre-seed corruption that pre-dates this fix (older version of the hook could
    // have persisted bad data). On the user's next drag, TanStack calls
    // `(prev) => ({ ...prev, [id]: nextWidth })`. Without the prev-sanitisation
    // wrapper, the bad entry would survive the spread and get re-persisted,
    // lingering forever. The wrapper hands the updater a clean prev so it dies
    // on the next write.
    seedRawWidths('exp-1', 'ds-1', { inputs: 400, source: -10 });

    const { result } = renderHook(() => usePersistedTableColumnWidths({ experimentId: 'exp-1', datasetId: 'ds-1' }));

    act(() => {
      result.current.setColumnSizing((prev) => ({ ...prev, expectations: 500 }));
    });

    expect(readRawWidths('exp-1', 'ds-1')).toEqual({ inputs: 400, expectations: 500 });
  });
});
