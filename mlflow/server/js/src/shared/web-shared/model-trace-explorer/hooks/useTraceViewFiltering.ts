import { useMemo } from 'react';

// @ts-expect-error jsonpath-plus does not ship type declarations
import { JSONPath } from 'jsonpath-plus';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { SpanFilter, TraceView } from './useTraceViews';

/**
 * Checks whether a span matches the given span filter criteria.
 * A span matches if all non-null filter fields match.
 */
export function spanMatchesFilter(span: ModelTraceSpanNode, filter: SpanFilter | null | undefined): boolean {
  if (!filter) return true;

  if (filter.span_name && String(span.title) !== filter.span_name) {
    return false;
  }

  if (filter.span_type) {
    const attrs = span.attributes;
    const spanType =
      span.type ?? (attrs && !Array.isArray(attrs) ? (attrs as Record<string, unknown>)['mlflow.spanType'] : undefined);
    if (spanType && String(spanType).toUpperCase() !== filter.span_type.toUpperCase()) {
      return false;
    }
  }

  if (filter.attribute_key) {
    const attrs = span.attributes;
    if (!attrs || Array.isArray(attrs)) {
      return false;
    }
    const attrValue = (attrs as Record<string, unknown>)[filter.attribute_key];
    if (attrValue === undefined) {
      return false;
    }
    if (filter.attribute_value && String(attrValue) !== filter.attribute_value) {
      return false;
    }
  }

  return true;
}

/**
 * Applies a JSONPath expression to a JSON string and returns the extracted value.
 * Falls back to the original data if the path is invalid or yields no results.
 */
export function applyJsonPath(data: string, jsonPath: string | null | undefined): string {
  if (!jsonPath) return data;
  try {
    const parsed = JSON.parse(data);
    const results = JSONPath({ path: jsonPath, json: parsed });
    if (!results || results.length === 0) return data;
    const extracted = results
      .filter((r: unknown) => r != null)
      .map((r: unknown) => (typeof r === 'object' ? JSON.stringify(r, null, 2) : String(r)))
      .join('\n');
    return extracted || data;
  } catch {
    return data;
  }
}

/**
 * Returns a set of span keys that match the active trace view's span filter.
 * When no view is active, returns null (meaning all spans match).
 */
export function useTraceViewSpanMatches(
  allNodes: ModelTraceSpanNode[],
  activeView: TraceView | null,
): Set<string | number> | null {
  return useMemo(() => {
    if (!activeView?.span_filter) return null;

    const matches = new Set<string | number>();

    const walk = (nodes: ModelTraceSpanNode[]) => {
      for (const node of nodes) {
        if (spanMatchesFilter(node, activeView.span_filter)) {
          matches.add(node.key);
        }
        if (node.children) {
          walk(node.children);
        }
      }
    };

    walk(allNodes);
    return matches;
  }, [allNodes, activeView]);
}
