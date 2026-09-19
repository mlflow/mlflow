import { useRef } from 'react';

/**
 * Pure page-token stack backing prev/next pagination over a cursor API.
 *
 * The search API is cursor-based: each page returns a `next_page_token` that fetches the *next*
 * page. To offer prev/next nav we remember, per query, the token that fetches each page:
 * `stack[i]` is the token for page `i + 1` (1-based pages). `stack[0]` is always `undefined`
 * (page 1 needs no token). After page N resolves we record its `next_page_token` into
 * `stack[N]` — so once page N has been seen, page N+1 becomes reachable.
 *
 * A `null` recorded next-token marks a known last page (distinct from `undefined` = "unknown /
 * not yet fetched"). This lets `hasNext` disable the Next button on the last page without a total
 * count.
 *
 * Pure and framework-free so it can be unit-tested without React. `useTraceTokenCache` wraps one
 * instance in a ref for the React layer.
 */
export class PageTokenStack {
  /** `slots[i]` = token to fetch page `i+1`. `slots[0]` = undefined (page 1). `null` = last page marker. */
  private slots: (string | null | undefined)[] = [undefined];
  private key = '';

  /** The cache key this stack is currently valid for. */
  getKey(): string {
    return this.key;
  }

  /**
   * Clear the stack when the query identity (locations/filter/orderBy/pageSize) changes. A cursor
   * from a previous query would point into a different result set. No-op when the key is unchanged.
   */
  resetIfKeyChanged(nextKey: string): void {
    if (nextKey !== this.key) {
      this.key = nextKey;
      this.slots = [undefined];
    }
  }

  /**
   * Token that fetches `page` (1-based), or `undefined` if unknown. Page 1 is always reachable
   * (returns `undefined`, meaning "no token"). For deeper pages this returns the recorded cursor,
   * or `undefined` when that page hasn't been made reachable yet.
   */
  getTokenForPage(page: number): string | undefined {
    if (page <= 1) {
      return undefined;
    }
    const slot = this.slots[page - 1];
    // `null` (last-page marker) and `undefined` (unknown) both mean "no usable token".
    return slot ?? undefined;
  }

  /**
   * True once `page` is reachable — page 1 always is, and a deeper page is reachable only when we
   * hold a real (string) token for it. A `null` slot (last-page marker on the page before) means
   * that page does not exist, so it is NOT known.
   */
  isPageKnown(page: number): boolean {
    if (page <= 1) {
      return true;
    }
    return typeof this.slots[page - 1] === 'string';
  }

  /**
   * Record the `next_page_token` returned when `page` (1-based) resolved. Pass `undefined`/`null`
   * (no next token) to mark `page` as the last page. Idempotent for a given (page, token).
   *
   * An empty-string token is normalized to `null`: proto3 serializes an absent `next_page_token` as
   * `""`, so an empty string is "no cursor", not a real one — treating it as a string would leave a
   * phantom next page that fetches the same rows forever.
   */
  recordNextToken(page: number, nextToken: string | undefined | null): void {
    // `slots[page]` fetches page `page+1`. A falsy token (null/undefined/'') records a definitive last page.
    this.slots[page] = nextToken || null;
  }

  /**
   * Whether a next page exists after `page`, derivable from the stack alone (no total needed).
   * Known-true when we hold a real token for `page+1`; known-false when `page` was marked last.
   * When `page` itself hasn't resolved yet we optimistically return `true` so Next stays enabled
   * until the current page resolves and tells us otherwise.
   */
  hasNext(page: number): boolean {
    const slot = this.slots[page];
    if (slot === null) {
      return false;
    }
    if (typeof slot === 'string') {
      return true;
    }
    // Unknown: current page not resolved yet — keep Next available (it'll disable on resolve).
    return true;
  }

  /** Whether a previous page exists. Purely positional. */
  hasPrev(page: number): boolean {
    return page > 1;
  }
}

export interface TraceTokenCache {
  getTokenForPage: (page: number) => string | undefined;
  isPageKnown: (page: number) => boolean;
  recordNextToken: (page: number, nextToken: string | undefined | null) => void;
  resetIfKeyChanged: (nextKey: string) => void;
  hasNext: (page: number) => boolean;
  hasPrev: (page: number) => boolean;
}

/**
 * React wrapper around a single `PageTokenStack` held in a ref. The stack is intentionally NOT
 * React state — recording a token must not trigger a re-render (the query result already does),
 * and the stack is read imperatively at fetch time.
 */
export const useTraceTokenCache = (): TraceTokenCache => {
  const stackRef = useRef<PageTokenStack | null>(null);
  if (stackRef.current === null) {
    stackRef.current = new PageTokenStack();
  }
  const stack = stackRef.current;

  return {
    getTokenForPage: (page) => stack.getTokenForPage(page),
    isPageKnown: (page) => stack.isPageKnown(page),
    recordNextToken: (page, nextToken) => stack.recordNextToken(page, nextToken),
    resetIfKeyChanged: (nextKey) => stack.resetIfKeyChanged(nextKey),
    hasNext: (page) => stack.hasNext(page),
    hasPrev: (page) => stack.hasPrev(page),
  };
};
