import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, renderHook } from '@testing-library/react';
import type { MutableRefObject } from 'react';
import { useTableHorizontalScrollAnchor } from './useTableHorizontalScrollAnchor';

const rect = (left: number, right: number): DOMRect =>
  ({ left, right, x: left, width: right - left, top: 0, bottom: 20, y: 0, height: 20, toJSON: () => ({}) }) as DOMRect;

let resizeCallback: ResizeObserverCallback;
const originalResizeObserver = globalThis.ResizeObserver;

beforeEach(() => {
  globalThis.ResizeObserver = class {
    constructor(callback: ResizeObserverCallback) {
      resizeCallback = callback;
    }
    observe = jest.fn();
    unobserve = jest.fn();
    disconnect = jest.fn();
  };
});

afterEach(() => {
  globalThis.ResizeObserver = originalResizeObserver;
});

describe('useTableHorizontalScrollAnchor', () => {
  test('keeps the leftmost visible header in place when rendered columns change', () => {
    const container = document.createElement('div');
    const scroller = document.createElement('div');
    const hiddenHeader = document.createElement('div');
    const anchorHeader = document.createElement('div');
    scroller.setAttribute('role', 'table');
    hiddenHeader.setAttribute('role', 'columnheader');
    anchorHeader.setAttribute('role', 'columnheader');
    scroller.append(hiddenHeader, anchorHeader);
    container.append(scroller);
    scroller.scrollLeft = 100;
    scroller.getBoundingClientRect = () => rect(0, 500);
    hiddenHeader.getBoundingClientRect = () => rect(-120, -20);
    let anchorLeft = -20;
    anchorHeader.getBoundingClientRect = () => rect(anchorLeft, anchorLeft + 100);
    const containerRef: MutableRefObject<HTMLDivElement> = { current: container };

    const { rerender } = renderHook(({ columnsKey }) => useTableHorizontalScrollAnchor(containerRef, columnsKey), {
      initialProps: { columnsKey: 'before' },
    });
    fireEvent.scroll(scroller);

    anchorLeft = 30;
    rerender({ columnsKey: 'after' });

    expect(scroller.scrollLeft).toBe(150);
  });

  test('does not adjust a table that has not been scrolled horizontally', () => {
    const container = document.createElement('div');
    const scroller = document.createElement('div');
    const header = document.createElement('div');
    scroller.setAttribute('role', 'table');
    header.setAttribute('role', 'columnheader');
    scroller.append(header);
    container.append(scroller);
    scroller.getBoundingClientRect = () => rect(0, 500);
    let headerLeft = 0;
    header.getBoundingClientRect = () => rect(headerLeft, headerLeft + 100);
    const containerRef: MutableRefObject<HTMLDivElement> = { current: container };

    const { rerender } = renderHook(({ columnsKey }) => useTableHorizontalScrollAnchor(containerRef, columnsKey), {
      initialProps: { columnsKey: 'before' },
    });
    fireEvent.scroll(scroller);

    headerLeft = 50;
    rerender({ columnsKey: 'after' });

    expect(scroller.scrollLeft).toBe(0);
  });

  test('refreshes the stored offset after a column resize', () => {
    const container = document.createElement('div');
    const scroller = document.createElement('div');
    const anchorHeader = document.createElement('div');
    scroller.setAttribute('role', 'table');
    anchorHeader.setAttribute('role', 'columnheader');
    scroller.append(anchorHeader);
    container.append(scroller);
    scroller.scrollLeft = 100;
    scroller.getBoundingClientRect = () => rect(0, 500);
    let anchorLeft = -20;
    anchorHeader.getBoundingClientRect = () => rect(anchorLeft, anchorLeft + 100);
    const containerRef: MutableRefObject<HTMLDivElement> = { current: container };

    const { rerender } = renderHook(({ columnsKey }) => useTableHorizontalScrollAnchor(containerRef, columnsKey), {
      initialProps: { columnsKey: 'before' },
    });
    fireEvent.scroll(scroller);

    anchorLeft = 30;
    resizeCallback([], {} as ResizeObserver);
    anchorLeft = 80;
    rerender({ columnsKey: 'after' });

    expect(scroller.scrollLeft).toBe(150);
  });

  test('ignores vertical-only scroll events', () => {
    const container = document.createElement('div');
    const scroller = document.createElement('div');
    const anchorHeader = document.createElement('div');
    scroller.setAttribute('role', 'table');
    anchorHeader.setAttribute('role', 'columnheader');
    scroller.append(anchorHeader);
    container.append(scroller);
    scroller.scrollLeft = 100;
    scroller.getBoundingClientRect = () => rect(0, 500);
    const getAnchorRect = jest.fn(() => rect(-20, 80));
    anchorHeader.getBoundingClientRect = getAnchorRect;
    const containerRef: MutableRefObject<HTMLDivElement> = { current: container };

    renderHook(() => useTableHorizontalScrollAnchor(containerRef, 'columns'));
    const measurementsAfterMount = getAnchorRect.mock.calls.length;
    fireEvent.scroll(scroller);

    expect(getAnchorRect).toHaveBeenCalledTimes(measurementsAfterMount);
  });
});
