import React from 'react';
import { useSyncExternalStore } from 'use-sync-external-store/shim';

function useMatchesMediaQuery(queryList: MediaQueryList) {
  return useSyncExternalStore(
    React.useCallback(
      (callback) => {
        queryList.addEventListener('change', callback);
        return () => {
          queryList.removeEventListener('change', callback);
        };
      },
      [queryList],
    ),
    () => queryList.matches,
  );
}

/**
 * React hook that listens for changes to a [media query][media-query]. Uses
 * [`window.matchMedia()`][match-media] under-the-hood.
 *
 * [media-query]: https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_media_queries/Using_media_queries
 * [match-media]: https://developer.mozilla.org/en-US/docs/Web/API/Window/matchMedia
 *
 * @usage
 *
 * ```tsx
 * function FancyButton() {
 *   const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion)');
 *   return prefersReducedMotion ? <Button /> : <DancingButton />;
 * }
 * ```
 *
 * > **Note**
 * > The vast majority of use-cases can (and should) use vanilla CSS media
 * > queries instead of this hook — which will cause a re-render when the match
 * > state changes. Usage of this hook should be reserved for use-cases where
 * > CSS cannot be used instead.
 * >
 * > ```tsx
 * > <Button css={{ 'not (prefers-reduced-motion)': { animation: … } }} />
 * > ```
 */
export function useMediaQuery(query: string) {
  // Note: a new MediaQueryList is created with every _usage_ of this hook.
  // It's probably cheap to create many instances of MediaQueryList, and
  // garbage collection will still clean up as expected, but consider using a
  // [weak cache](https://github.com/tc39/proposal-weakrefs#weak-caches) to
  // reuse MediaQueryLists where possible if performance is impacted.
  const queryList = React.useMemo(() => window.matchMedia(query), [query]);
  return useMatchesMediaQuery(queryList);
}
