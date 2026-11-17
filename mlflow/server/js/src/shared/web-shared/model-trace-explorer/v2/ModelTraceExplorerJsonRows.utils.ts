import { isObject } from 'lodash';

// Mirrors CollapsibleJsonViewer, whose nesting cutoff and line numbering these rows reproduce.
export const JSON_ROWS_MAX_DEPTH = 100;
export const LARGE_JSON_MIN_CHARS = 100_000;
export const DEFAULT_JSON_ROWS_EXPAND_DEPTH = 5;

export type JsonRow =
  | { kind: 'primitive'; depth: number; nodeKey?: string; value: unknown; isLast: boolean; line: number }
  | {
      kind: 'open';
      depth: number;
      nodeKey?: string;
      container: WeakKey;
      isArray: boolean;
      entryCount: number;
      collapsed: boolean;
      isLast: boolean;
      line: number;
    }
  | { kind: 'close'; depth: number; isArray: boolean; isLast: boolean; line: number }
  | { kind: 'maxDepth'; depth: number; line: number };

const lineCounts = new WeakMap<WeakKey, Map<number, number>>();

/** Lines a value spans when fully expanded: one per primitive, plus an opening and closing line per container. */
const countJsonLines = (value: unknown, depth: number): number => {
  if (depth > JSON_ROWS_MAX_DEPTH || !isObject(value)) {
    return 1;
  }
  const cached = lineCounts.get(value)?.get(depth);
  if (cached !== undefined) {
    return cached;
  }
  let lines = 2;
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index++) {
      lines += countJsonLines(value[index], depth + 1);
    }
  } else {
    for (const key of Object.keys(value)) {
      lines += countJsonLines((value as Record<string, unknown>)[key], depth + 1);
    }
  }
  const depths = lineCounts.get(value) ?? new Map<number, number>();
  depths.set(depth, lines);
  lineCounts.set(value, depths);
  return lines;
};

export interface JsonRowsResult {
  rows: JsonRow[];
  /** True when `maxRows` cut the rows short. */
  isTruncated: boolean;
}

/**
 * Flattens a JSON value into the lines CollapsibleJsonViewer would draw, walking only expanded containers.
 * Line numbers count collapsed content as if it were expanded, as that viewer does. With `maxRows`, the walk
 * bounds rendered rows. Logical line counting can still visit entire collapsed subtrees.
 */
export const getJsonRows = (
  root: unknown,
  isExpanded: (container: WeakKey, depth: number) => boolean,
  maxRows: number = Infinity,
): JsonRowsResult => {
  const rows: JsonRow[] = [];
  let nextLine = 1;
  let isTruncated = false;

  const visit = (value: unknown, depth: number, nodeKey: string | undefined, isLast: boolean): void => {
    if (rows.length >= maxRows) {
      isTruncated = true;
      return;
    }
    if (depth > JSON_ROWS_MAX_DEPTH) {
      rows.push({ kind: 'maxDepth', depth, line: nextLine++ });
      return;
    }
    if (!isObject(value)) {
      rows.push({ kind: 'primitive', depth, nodeKey, value, isLast, line: nextLine++ });
      return;
    }

    const isArray = Array.isArray(value);
    const keys = isArray ? undefined : Object.keys(value);
    const entryCount = isArray ? value.length : (keys?.length ?? 0);
    if (!isExpanded(value, depth)) {
      const lines = countJsonLines(value, depth);
      // a collapsed container is labelled with its closing line, as in CollapsibleJsonViewer
      rows.push({
        kind: 'open',
        depth,
        nodeKey,
        container: value,
        isArray,
        entryCount,
        collapsed: true,
        isLast,
        line: nextLine + lines - 1,
      });
      nextLine += lines;
      return;
    }

    rows.push({
      kind: 'open',
      depth,
      nodeKey,
      container: value,
      isArray,
      entryCount,
      collapsed: false,
      isLast,
      line: nextLine++,
    });
    for (let index = 0; index < entryCount; index++) {
      if (isTruncated) break;
      const childIsLast = index === entryCount - 1;
      if (isArray) {
        visit(value[index], depth + 1, undefined, childIsLast);
      } else {
        const key = (keys as string[])[index];
        visit((value as Record<string, unknown>)[key], depth + 1, key, childIsLast);
      }
    }
    if (isTruncated) {
      return;
    }
    const closingLine = nextLine++;
    // an empty container closes on its opening row
    if (entryCount > 0) {
      if (rows.length >= maxRows) {
        isTruncated = true;
        return;
      }
      rows.push({ kind: 'close', depth, isArray, isLast, line: closingLine });
    }
  };

  visit(root, 0, undefined, true);
  return { rows, isTruncated };
};

/** Expand depth for a payload of the given JSON length; deep defaults on multi-MB payloads expand most of it. */
export const getJsonRowsExpandDepth = (jsonLength: number): number => {
  if (jsonLength > 1_000_000) {
    return 1;
  }
  if (jsonLength >= LARGE_JSON_MIN_CHARS) {
    return 2;
  }
  return DEFAULT_JSON_ROWS_EXPAND_DEPTH;
};
