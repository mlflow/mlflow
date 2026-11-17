import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { getMatchesFromSpan } from '../ModelTraceExplorer.utils';

/** Collects payload matches between browser tasks, returning a cleanup function for obsolete searches. */
export function collectModelTraceSearchMatches(
  roots: ModelTraceSpanNode[],
  filter: string,
  onComplete: (matches: Map<ModelTraceSpanNode, SearchMatch[]>) => void,
): () => void {
  const pending = [...roots];
  const matches = new Map<ModelTraceSpanNode, SearchMatch[]>();
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const collectBatch = () => {
    if (cancelled) return;
    const deadline = performance.now() + 8;
    // shortcut: yield between spans; slice within a span if individual payloads still block input.
    while (pending.length > 0) {
      const span = pending.pop();
      if (span && !matches.has(span)) {
        matches.set(span, getMatchesFromSpan(span, filter));
        for (const child of span.children ?? []) pending.push(child);
      }
      if (performance.now() >= deadline) break;
    }
    if (pending.length > 0) {
      timer = setTimeout(collectBatch, 0);
    } else {
      onComplete(matches);
    }
  };
  collectBatch();
  return () => {
    cancelled = true;
    clearTimeout(timer);
  };
}
