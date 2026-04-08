import { useMemo } from 'react';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { decodeSpanId } from '../ModelTraceExplorer.utils';
import { getTimelineTreeNodesList } from '../timeline-tree/TimelineTree.utils';
import type { SpanRange, SpanSelector, TraceView } from './useTraceViews';

// Lazy-load jsonpath-plus to avoid top-level import failures
let JSONPath: ((opts: { path: string; json: unknown }) => unknown[]) | null = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  JSONPath = require('jsonpath-plus').JSONPath;
} catch {
  // jsonpath-plus not available — applyJsonPath will return raw data
}

/**
 * Checks whether a span matches the given span selector criteria.
 * A span matches if all non-null selector fields match.
 */
export function spanMatchesSelector(span: ModelTraceSpanNode, selector: SpanSelector | null | undefined): boolean {
  if (!selector) return true;

  if (selector.span_id) {
    // Selector span_id may be base64 while node.key is hex-decoded; normalize both
    const nodeKey = String(span.key);
    const selectorId = decodeSpanId(selector.span_id, true);
    if (nodeKey !== selectorId && nodeKey !== selector.span_id) {
      return false;
    }
  }

  if (selector.span_name && String(span.title) !== selector.span_name) {
    return false;
  }

  if (selector.span_type) {
    const attrs = span.attributes;
    const spanType =
      span.type ?? (attrs && !Array.isArray(attrs) ? (attrs as Record<string, unknown>)['mlflow.spanType'] : undefined);
    if (spanType && String(spanType).toUpperCase() !== selector.span_type.toUpperCase()) {
      return false;
    }
  }

  if (selector.attribute_key) {
    const attrs = span.attributes;
    if (!attrs || Array.isArray(attrs)) {
      return false;
    }
    const attrValue = (attrs as Record<string, unknown>)[selector.attribute_key];
    if (attrValue === undefined) {
      return false;
    }
    if (selector.attribute_value && String(attrValue) !== selector.attribute_value) {
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
  if (!jsonPath || !JSONPath) return data;
  try {
    const parsed = JSON.parse(data);
    const results = JSONPath({ path: jsonPath, json: parsed });
    if (!results || results.length === 0) return data;
    // Filter nulls
    const filtered = results.filter((r: unknown) => r != null);
    if (filtered.length === 0) return data;
    // If single result, JSON-serialize it so downstream components can parse it.
    // ModelTraceExplorerCodeSnippet expects valid JSON strings.
    if (filtered.length === 1) {
      return JSON.stringify(filtered[0], null, 2);
    }
    // Multiple results: serialize as a JSON array
    return JSON.stringify(filtered, null, 2);
  } catch {
    return data;
  }
}

/**
 * Applies a JSONPath expression to a raw data object (e.g. span inputs/outputs)
 * and returns the extracted result. Unlike `applyJsonPath` which operates on
 * individual serialized values, this operates on the whole object so that
 * paths like `$.reasoning` correctly select a top-level key from the object.
 *
 * Returns the original data unchanged if the path is invalid, yields no results,
 * or jsonpath-plus is unavailable.
 */
export function applyJsonPathToObject(data: unknown, jsonPath: string | null | undefined): unknown {
  if (!jsonPath || !JSONPath || data == null) return data;
  try {
    const results = JSONPath({ path: jsonPath, json: data });
    if (!results || results.length === 0) return data;
    const filtered = results.filter((r: unknown) => r != null);
    if (filtered.length === 0) return data;
    if (filtered.length === 1) return filtered[0];
    return filtered;
  } catch {
    return data;
  }
}

/**
 * Checks whether a span falls within a range in tree-order.
 * For single-span ranges (no to_selector), checks if the span matches from_selector.
 * For multi-span ranges, checks if the span appears between from and to in the flat node list.
 */
export function isSpanInRange(
  node: ModelTraceSpanNode,
  flatNodes: ModelTraceSpanNode[],
  range: SpanRange,
): boolean {
  if (!range.to_selector) {
    return spanMatchesSelector(node, range.from_selector);
  }
  let inRange = false;
  for (const n of flatNodes) {
    if (spanMatchesSelector(n, range.from_selector)) inRange = true;
    if (inRange && n.key === node.key) return true;
    if (spanMatchesSelector(n, range.to_selector) && inRange) break;
  }
  return false;
}

export interface SpanViewRangeInfo {
  rangeIdx: number;
  isFirstInRange: boolean;
}

/**
 * Returns a set of span keys that match the active trace view's span ranges,
 * and a map from span key to range info (index, whether it's the first span in range).
 * When no view is active or the view has no ranges, returns null for both.
 */
export function useTraceViewSpanMatches(
  allNodes: ModelTraceSpanNode[],
  activeView: TraceView | null,
): { matchedKeys: Set<string | number> | null; rangeMap: Map<string | number, SpanViewRangeInfo> | null } {
  return useMemo(() => {
    if (!activeView?.ranges || activeView.ranges.length === 0) {
      return { matchedKeys: null, rangeMap: null };
    }

    const matchedKeys = new Set<string | number>();
    const rangeMap = new Map<string | number, SpanViewRangeInfo>();
    const flatNodes = getTimelineTreeNodesList(allNodes);

    // For each range, find the first matching span to mark as isFirstInRange
    const rangeFirstKeys = new Set<string>();
    for (let rangeIdx = 0; rangeIdx < activeView.ranges.length; rangeIdx++) {
      const range = activeView.ranges[rangeIdx];
      const firstMatch = flatNodes.find((n) => spanMatchesSelector(n, range.from_selector));
      if (firstMatch) rangeFirstKeys.add(`${rangeIdx}:${firstMatch.key}`);
    }

    for (const node of flatNodes) {
      for (let rangeIdx = 0; rangeIdx < activeView.ranges.length; rangeIdx++) {
        if (isSpanInRange(node, flatNodes, activeView.ranges[rangeIdx])) {
          matchedKeys.add(node.key);
          rangeMap.set(node.key, {
            rangeIdx,
            isFirstInRange: rangeFirstKeys.has(`${rangeIdx}:${node.key}`),
          });
          break;
        }
      }
    }

    return { matchedKeys, rangeMap };
  }, [allNodes, activeView]);
}
