import { useCallback } from 'react';

const INFINITE_SCROLL_BOTTOM_OFFSET = 200;

/**
 * Util function to fetch next page when user scrolls to the end of a scrollable container
 */
export const useInfiniteScrollFetch = ({
  isFetching,
  hasNextPage,
  fetchNextPage,
}: {
  isFetching: boolean;
  hasNextPage: boolean;
  fetchNextPage: () => void;
}) => {
  return useCallback(
    (containerRefElement?: HTMLDivElement | null) => {
      if (containerRefElement) {
        const { scrollHeight, scrollTop, clientHeight } = containerRefElement;
        if (scrollHeight - scrollTop - clientHeight < INFINITE_SCROLL_BOTTOM_OFFSET && !isFetching && hasNextPage) {
          fetchNextPage();
        }
      }
    },
    [fetchNextPage, isFetching, hasNextPage],
  );
};
