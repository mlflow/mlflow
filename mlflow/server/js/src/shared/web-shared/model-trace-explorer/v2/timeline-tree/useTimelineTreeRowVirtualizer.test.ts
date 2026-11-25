import { describe, expect, test } from '@jest/globals';

import { mergePinnedRowIndex } from './useTimelineTreeRowVirtualizer';

describe('mergePinnedRowIndex', () => {
  const range = { startIndex: 3, endIndex: 5, overscan: 1, count: 10 };

  test('adds an off-screen selected row once in order', () => {
    expect(mergePinnedRowIndex(range, 9)).toEqual([2, 3, 4, 5, 6, 9]);
  });

  test('does not duplicate a visible or missing selected row', () => {
    expect(mergePinnedRowIndex(range, 4)).toEqual([2, 3, 4, 5, 6]);
    expect(mergePinnedRowIndex(range, -1)).toEqual([2, 3, 4, 5, 6]);
  });
});
