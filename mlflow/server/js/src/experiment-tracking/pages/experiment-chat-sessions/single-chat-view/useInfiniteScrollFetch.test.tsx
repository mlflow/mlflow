import { describe, it, expect, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import type { RefObject, UIEvent } from 'react';
import { useInfiniteScrollFetch } from './useInfiniteScrollFetch';

const makeContainerRef = (overrides: Partial<HTMLDivElement>): RefObject<HTMLDivElement> => ({
  current: {
    scrollTop: 0,
    ...overrides,
  } as HTMLDivElement,
});

describe('useInfiniteScrollFetch', () => {
  it('auto-fetches on mount when the container is not tall enough to scroll', () => {
    const fetchNextPage = jest.fn();
    const containerRef = makeContainerRef({ scrollHeight: 400, clientHeight: 500 });

    renderHook(() =>
      useInfiniteScrollFetch({ containerRef, fetchNextPage, hasNextPage: true, isFetchingNextPage: false }),
    );

    expect(fetchNextPage).toHaveBeenCalledTimes(1);
  });

  it('does not auto-fetch on mount when the container already scrolls', () => {
    const fetchNextPage = jest.fn();
    const containerRef = makeContainerRef({ scrollHeight: 2000, clientHeight: 500 });

    renderHook(() =>
      useInfiniteScrollFetch({ containerRef, fetchNextPage, hasNextPage: true, isFetchingNextPage: false }),
    );

    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it('does not auto-fetch on mount when there is no next page', () => {
    const fetchNextPage = jest.fn();
    const containerRef = makeContainerRef({ scrollHeight: 400, clientHeight: 500 });

    renderHook(() =>
      useInfiniteScrollFetch({ containerRef, fetchNextPage, hasNextPage: false, isFetchingNextPage: false }),
    );

    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it('does not auto-fetch on mount while a page is already being fetched', () => {
    const fetchNextPage = jest.fn();
    const containerRef = makeContainerRef({ scrollHeight: 400, clientHeight: 500 });

    renderHook(() =>
      useInfiniteScrollFetch({ containerRef, fetchNextPage, hasNextPage: true, isFetchingNextPage: true }),
    );

    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it('fetches the next page when scrolled near the bottom', () => {
    const fetchNextPage = jest.fn();
    // Tall enough to scroll already, so the mount effect doesn't fire on its own.
    const containerRef = makeContainerRef({ scrollHeight: 2000, clientHeight: 500 });

    const { result } = renderHook(() =>
      useInfiniteScrollFetch({ containerRef, fetchNextPage, hasNextPage: true, isFetchingNextPage: false }),
    );

    result.current({
      currentTarget: { scrollHeight: 2000, clientHeight: 500, scrollTop: 1850 },
    } as UIEvent<HTMLDivElement>);

    expect(fetchNextPage).toHaveBeenCalledTimes(1);
  });

  it('does not fetch on scroll while far from the bottom', () => {
    const fetchNextPage = jest.fn();
    const containerRef = makeContainerRef({ scrollHeight: 2000, clientHeight: 500 });

    const { result } = renderHook(() =>
      useInfiniteScrollFetch({ containerRef, fetchNextPage, hasNextPage: true, isFetchingNextPage: false }),
    );

    result.current({
      currentTarget: { scrollHeight: 2000, clientHeight: 500, scrollTop: 100 },
    } as UIEvent<HTMLDivElement>);

    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it('does not fetch on scroll near the bottom when there is no next page', () => {
    const fetchNextPage = jest.fn();
    const containerRef = makeContainerRef({ scrollHeight: 2000, clientHeight: 500 });

    const { result } = renderHook(() =>
      useInfiniteScrollFetch({ containerRef, fetchNextPage, hasNextPage: false, isFetchingNextPage: false }),
    );

    result.current({
      currentTarget: { scrollHeight: 2000, clientHeight: 500, scrollTop: 1850 },
    } as UIEvent<HTMLDivElement>);

    expect(fetchNextPage).not.toHaveBeenCalled();
  });
});
