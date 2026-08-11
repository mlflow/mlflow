import { useCallback, useMemo } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
// Reuse the generic (branding-free) number-search-param helper from datasets-v2.
import { useNumberSearchParam } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets-v2/hooks/useNumberSearchParam';
import {
  DEFAULT_PAGE_SIZE,
  DEFAULT_SORT_COLUMN,
  DEFAULT_SORT_DIR,
  PAGE_SIZE_OPTIONS,
  isSortableTraceColumn,
  type PageSize,
  type SortDirection,
  type TraceColumnId,
} from '@databricks/web-shared/traces-table';

const Q_PARAM = 'q';
const PAGE_PARAM = 'page';
const PAGE_SIZE_PARAM = 'pageSize';
const SORT_PARAM = 'sort';
const DIR_PARAM = 'dir';
const TRACE_ID_PARAM = 'traceId';
// Repeatable param (like v3's `filter`): each value is `encodeURIComponent(key)=encodeURIComponent(value)`.
const TAG_PARAM = 'tag';

/** A single click-to-filter tag constraint (`tags.<key> = '<value>'`), persisted in the URL. */
export interface TagFilter {
  key: string;
  value: string;
}

/** Encode a (key, value) as one `tag` param value. Component-encoded so `:`/spaces/`.`/`=` survive. */
const encodeTagFilter = ({ key, value }: TagFilter): string =>
  `${encodeURIComponent(key)}=${encodeURIComponent(value)}`;

/** Parse one `tag` param value back into a filter, or `undefined` if it has no `key=value` shape. */
const decodeTagFilter = (raw: string): TagFilter | undefined => {
  const eq = raw.indexOf('=');
  if (eq < 0) {
    return undefined;
  }
  return { key: decodeURIComponent(raw.slice(0, eq)), value: decodeURIComponent(raw.slice(eq + 1)) };
};

const sameTagFilter = (a: TagFilter, b: TagFilter): boolean => a.key === b.key && a.value === b.value;

const isSortableColumnId = (value: string | null): value is TraceColumnId =>
  value !== null && isSortableTraceColumn(value);

const PAGE_SIZE_SET = new Set<number>(PAGE_SIZE_OPTIONS);
const toValidPageSize = (raw: string | null): PageSize => {
  const parsed = raw === null ? NaN : Number.parseInt(raw, 10);
  return PAGE_SIZE_SET.has(parsed) ? (parsed as PageSize) : DEFAULT_PAGE_SIZE;
};

export interface TracesV4UrlState {
  search: string;
  setSearch: (next: string) => void;
  /** 1-based page index. */
  pageIndex: number;
  setPageIndex: (next: number) => void;
  pageSize: PageSize;
  setPageSize: (next: PageSize) => void;
  /** Active sort column. Only `start_time`/`duration` are honored; anything else → default. */
  sort: TraceColumnId;
  dir: SortDirection;
  setSort: (column: TraceColumnId, direction: SortDirection) => void;
  /** Trace id whose detail drawer is open, or undefined. */
  traceId: string | undefined;
  setTraceId: (next: string | undefined) => void;
  /** Click-to-filter tag constraints, in URL order. */
  tagFilters: TagFilter[];
  /** Add a tag filter; toggles off if the identical (key, value) is already present. Resets `?page`. */
  addTagFilter: (key: string, value: string) => void;
  /** Replace the whole tag-filter set (deterministic; used for programmatic apply). Resets `?page`. */
  setTagFilters: (filters: TagFilter[]) => void;
  /** Remove a specific tag filter. Resets `?page`. */
  removeTagFilter: (key: string, value: string) => void;
  /** Clear all tag filters. Resets `?page`. */
  clearTagFilters: () => void;
}

/**
 * Owns the traces-v4 URL state: search query, 1-based page, page size, sort column + direction,
 * and the open trace-id.
 *
 * Any change that reframes the result set — search, sort, page size, or a tag filter — resets the
 * page back to 1. Applying a page cursor from the previous query against a re-filtered/re-sorted/
 * re-sized set would land the user on an empty or arbitrary page, so we drop `?page` on every such
 * change (mirrors `useDatasetRecordsUrlState`). Writing a default value removes its param to keep
 * URLs tidy.
 */
export const useTracesV4UrlState = (): TracesV4UrlState => {
  // OSS's `useSearchParams` is the raw react-router hook — it returns `[searchParams, setSearchParams]`
  // and has no read-selector overload (unlike the Databricks variant), so read each param off the
  // `searchParams` object directly.
  const [searchParams, setSearchParams] = useSearchParams();
  const search = searchParams.get(Q_PARAM) ?? '';
  const [pageIndex, setPageIndex] = useNumberSearchParam({ key: PAGE_PARAM, defaultValue: 1, min: 1 });
  const pageSizeRaw = searchParams.get(PAGE_SIZE_PARAM);
  const sortRaw = searchParams.get(SORT_PARAM);
  const dirRaw = searchParams.get(DIR_PARAM);
  const traceId = searchParams.get(TRACE_ID_PARAM) ?? undefined;
  // getAll → the repeatable `tag` values. Memoize on the serialized params: `getAll().map().filter()`
  // returns a fresh array every render, and consumers use `tagFilters` as an effect dependency (e.g.
  // the controller's clear-selection effect). Without a stable identity that effect would re-run on
  // every render and, in this case, wipe the bulk selection the instant it's made.
  const tagParamsKey = searchParams.getAll(TAG_PARAM).join('\n');
  const tagFilters = useMemo(
    () =>
      tagParamsKey
        .split('\n')
        .filter((value) => value !== '')
        .map(decodeTagFilter)
        .filter((filter): filter is TagFilter => filter !== undefined),
    [tagParamsKey],
  );

  const pageSize = toValidPageSize(pageSizeRaw);
  const sort: TraceColumnId = isSortableColumnId(sortRaw) ? sortRaw : DEFAULT_SORT_COLUMN;
  const dir: SortDirection = dirRaw === 'asc' ? 'asc' : dirRaw === 'desc' ? 'desc' : DEFAULT_SORT_DIR;

  const setSearch = useCallback(
    (next: string) => {
      setSearchParams((params) => {
        if (next) {
          params.set(Q_PARAM, next);
        } else {
          params.delete(Q_PARAM);
        }
        params.delete(PAGE_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  const setPageSize = useCallback(
    (next: PageSize) => {
      setSearchParams((params) => {
        if (next === DEFAULT_PAGE_SIZE) {
          params.delete(PAGE_SIZE_PARAM);
        } else {
          params.set(PAGE_SIZE_PARAM, String(next));
        }
        // Page size changes the slicing entirely — a page cursor from the old size is meaningless.
        params.delete(PAGE_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  const setSort = useCallback(
    (column: TraceColumnId, direction: SortDirection) => {
      setSearchParams((params) => {
        if (column === DEFAULT_SORT_COLUMN && direction === DEFAULT_SORT_DIR) {
          params.delete(SORT_PARAM);
          params.delete(DIR_PARAM);
        } else {
          params.set(SORT_PARAM, column);
          params.set(DIR_PARAM, direction);
        }
        params.delete(PAGE_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  const setTraceId = useCallback(
    (next: string | undefined) => {
      setSearchParams((params) => {
        if (next) {
          params.set(TRACE_ID_PARAM, next);
        } else {
          params.delete(TRACE_ID_PARAM);
        }
        return params;
      });
    },
    [setSearchParams],
  );

  // Rewrite the whole `tag` param set from a transform of the current list. Centralizes the
  // delete-all-then-re-append dance (URLSearchParams has no "replace all of key X") and the shared
  // `?page` reset that every tag-filter change needs.
  const rewriteTagFilters = useCallback(
    (transform: (current: TagFilter[]) => TagFilter[]) => {
      setSearchParams((params) => {
        const current = params
          .getAll(TAG_PARAM)
          .map(decodeTagFilter)
          .filter((filter): filter is TagFilter => filter !== undefined);
        params.delete(TAG_PARAM);
        for (const filter of transform(current)) {
          params.append(TAG_PARAM, encodeTagFilter(filter));
        }
        params.delete(PAGE_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  const addTagFilter = useCallback(
    (key: string, value: string) => {
      const next: TagFilter = { key, value };
      // Toggle: an identical (key, value) click removes it; otherwise append.
      rewriteTagFilters((current) =>
        current.some((filter) => sameTagFilter(filter, next))
          ? current.filter((filter) => !sameTagFilter(filter, next))
          : [...current, next],
      );
    },
    [rewriteTagFilters],
  );

  const removeTagFilter = useCallback(
    (key: string, value: string) => {
      const target: TagFilter = { key, value };
      rewriteTagFilters((current) => current.filter((filter) => !sameTagFilter(filter, target)));
    },
    [rewriteTagFilters],
  );

  const setTagFilters = useCallback((filters: TagFilter[]) => rewriteTagFilters(() => filters), [rewriteTagFilters]);

  const clearTagFilters = useCallback(() => rewriteTagFilters(() => []), [rewriteTagFilters]);

  return {
    search,
    setSearch,
    pageIndex,
    setPageIndex,
    pageSize,
    setPageSize,
    sort,
    dir,
    setSort,
    traceId,
    setTraceId,
    tagFilters,
    addTagFilter,
    setTagFilters,
    removeTagFilter,
    clearTagFilters,
  };
};
