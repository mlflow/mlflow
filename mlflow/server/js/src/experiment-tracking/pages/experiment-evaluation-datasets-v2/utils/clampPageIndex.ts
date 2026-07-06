/**
 * Clamp a 1-based `pageIndex` against the available result size. Returns the input unchanged
 * when it's already within range or when there are no items (so the effect doesn't fight an
 * empty-result render). Used by the records page to recover gracefully from stale URLs and
 * post-delete shrinkage that would otherwise land the user on a blank page.
 */
export const clampPageIndex = (pageIndex: number, totalItems: number, pageSize: number): number => {
  if (totalItems <= 0 || pageSize <= 0) return pageIndex;
  const lastValidPage = Math.max(1, Math.ceil(totalItems / pageSize));
  return pageIndex > lastValidPage ? lastValidPage : pageIndex;
};
