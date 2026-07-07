import { describe, expect, test } from '@jest/globals';
import { clampPageIndex } from './clampPageIndex';

describe('clampPageIndex', () => {
  test('returns the input when pageIndex is in range', () => {
    expect(clampPageIndex(2, 30, 10)).toBe(2);
    expect(clampPageIndex(3, 30, 10)).toBe(3);
    expect(clampPageIndex(1, 30, 10)).toBe(1);
  });

  test('clamps to the last valid page when pageIndex is past the end', () => {
    expect(clampPageIndex(5, 30, 10)).toBe(3);
    expect(clampPageIndex(99, 5, 10)).toBe(1);
  });

  test('handles partial last pages correctly', () => {
    // 23 items / 10 per page => 3 pages, last one partial.
    expect(clampPageIndex(5, 23, 10)).toBe(3);
    expect(clampPageIndex(3, 23, 10)).toBe(3);
  });

  test('leaves pageIndex alone when there are no items so we do not fight empty renders', () => {
    expect(clampPageIndex(5, 0, 10)).toBe(5);
    expect(clampPageIndex(1, 0, 10)).toBe(1);
  });

  test('treats non-positive pageSize as a no-op', () => {
    expect(clampPageIndex(3, 30, 0)).toBe(3);
    expect(clampPageIndex(3, 30, -1)).toBe(3);
  });
});
