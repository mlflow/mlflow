import type { RefObject } from 'react';
import { useCallback, useEffect } from 'react';

// Mirrors GenAiTracesTableBody's infinite-scroll trigger (genai-traces-table/GenAiTracesTableBody.tsx).
const FETCH_NEAR_BOTTOM_THRESHOLD_PX = 200;

/**
 * Fetches the next page of an infinite-paginated query when a scrollable container nears its
 * bottom, or immediately if the container isn't tall enough to scroll in the first place
 * (e.g. a short conversation on a tall viewport).
 */
export function useInfiniteScrollFetch({
  containerRef,
  fetchNextPage,
  hasNextPage,
  isFetchingNextPage,
}: {
  containerRef: RefObject<HTMLDivElement>;
  fetchNextPage?: () => void;
  hasNextPage?: boolean;
  isFetchingNextPage?: boolean;
}) {
  const onScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      if (!fetchNextPage || !hasNextPage || isFetchingNextPage) return;
      const { scrollHeight, scrollTop, clientHeight } = e.currentTarget;
      if (scrollHeight - scrollTop - clientHeight < FETCH_NEAR_BOTTOM_THRESHOLD_PX) {
        fetchNextPage();
      }
    },
    [fetchNextPage, hasNextPage, isFetchingNextPage],
  );

  // No dependency array: re-checks after every render (guarded by hasNextPage/isFetchingNextPage
  // below, so this is cheap), since there's no single value to depend on that changes exactly
  // when newly-fetched content grows the container's scrollHeight.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !fetchNextPage || !hasNextPage || isFetchingNextPage) return;
    if (container.scrollHeight <= container.clientHeight) {
      fetchNextPage();
    }
  });

  return onScroll;
}
