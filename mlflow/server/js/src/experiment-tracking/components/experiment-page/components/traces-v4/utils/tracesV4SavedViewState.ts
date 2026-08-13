import { type TraceColumnId } from '@databricks/web-shared/traces-table';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';

/**
 * Pure serialization for V4 saved views.
 *
 * Unlike the V3 tab (whose columns/sort live in localStorage, forcing a live-state bridge + a
 * React-state preview overlay), the V4 tab is URL-first: search, sort, page size, tag filters and
 * the time range all already live in the URL. So a V4 "view" is essentially a snapshot of the URL
 * search string (minus the transient page/traceId/share-key params), plus the one piece of view
 * state that ISN'T in the URL — column visibility — carried as a `cols` param. Applying a view is
 * then just a navigation to the rebuilt query; the URL itself is the applied view, and the `cols`
 * param doubles as the preview overlay (present only while a shared view is applied).
 */

// Distinct from the V3 prefix (`mlflow.traceViewState.`) and the runs prefix
// (`mlflow.sharedViewState.`): V3's `sort` wire format (`key::type::asc`) is incompatible with V4's
// separate `sort`+`dir` params, so a V4 list must never show a V3 or runs view (and vice-versa).
export const TRACE_V4_SAVED_VIEW_TAG_PREFIX = 'mlflow.tracesV4ViewState.';
export const TRACE_V4_SHARE_URL_PARAM_KEY = 'traceViewShareKey';

// The `cols` param carries column visibility (the only view state not otherwise in the URL) and, by
// its presence, marks a live preview of a shared view.
export const TRACE_V4_COLS_PARAM_KEY = 'cols';

const COLUMNS_SEPARATOR = ',';

// The URL view params that make up a V4 saved view. `tag` is repeatable (one param per filter).
// Deliberately EXCLUDES the transient params `page`, `traceId`, and the share key itself.
const SINGLE_VALUE_KEYS = ['q', 'pageSize', 'sort', 'dir', 'startTimeLabel', 'startTime', 'endTime'] as const;
const MULTI_VALUE_KEYS = ['tag'] as const;

export interface CapturedV4ViewState {
  single: Partial<Record<(typeof SINGLE_VALUE_KEYS)[number] | typeof TRACE_V4_COLS_PARAM_KEY, string>>;
  multi: Partial<Record<(typeof MULTI_VALUE_KEYS)[number], string[]>>;
}

export const getTraceV4SavedViewTagKey = (id: string): string => `${TRACE_V4_SAVED_VIEW_TAG_PREFIX}${id}`;

export const getTraceV4SavedViewIdFromTagKey = (key: string): string | null => {
  if (!key.startsWith(TRACE_V4_SAVED_VIEW_TAG_PREFIX)) {
    return null;
  }
  // A key that is exactly the prefix (no id) yields an empty id, which would collide across tags;
  // treat it as not a saved-view key (mirrors the V3 / runs helpers).
  const id = key.slice(TRACE_V4_SAVED_VIEW_TAG_PREFIX.length);
  return id === '' ? null : id;
};

/**
 * Whether the URL carries any serialized view state. A genuine share link built by
 * {@link buildV4ViewQuery} always includes at least one of these; a bare or garbage share key has
 * none. Derived from the same key lists as the capture/build path so the two can't drift — callers
 * use this to decide whether a share key is actually previewing a view. Presence is `!== null`, not
 * truthiness, so an empty-string value (e.g. `q=`) still counts as a captured value.
 */
export const urlHasCapturedV4ViewState = (params: URLSearchParams): boolean =>
  SINGLE_VALUE_KEYS.some((key) => params.get(key) !== null) ||
  MULTI_VALUE_KEYS.some((key) => params.getAll(key).length > 0) ||
  params.get(TRACE_V4_COLS_PARAM_KEY) !== null;

/**
 * Capture the current view: the whitelisted URL params plus the live visible columns (which live in
 * localStorage, not the URL, so they're passed in rather than read from `params`). The incoming
 * URL's own `cols` / share key are intentionally ignored so opening view A then saving view B never
 * leaks A's columns or id into B.
 */
export const captureV4ViewState = (
  params: URLSearchParams,
  visibleColumns: readonly TraceColumnId[],
): CapturedV4ViewState => {
  const single: CapturedV4ViewState['single'] = {};
  SINGLE_VALUE_KEYS.forEach((key) => {
    const value = params.get(key);
    if (value !== null) {
      single[key] = value;
    }
  });
  if (visibleColumns.length > 0) {
    single[TRACE_V4_COLS_PARAM_KEY] = visibleColumns.join(COLUMNS_SEPARATOR);
  }
  const multi: CapturedV4ViewState['multi'] = {};
  MULTI_VALUE_KEYS.forEach((key) => {
    const values = params.getAll(key);
    if (values.length > 0) {
      multi[key] = values;
    }
  });
  return { single, multi };
};

/** Rebuild a URL query string from a captured view + the view's id (as the share key). */
export const buildV4ViewQuery = (state: CapturedV4ViewState, viewId: string): string => {
  const params = new URLSearchParams();
  Object.entries(state.single ?? {}).forEach(([key, value]) => {
    if (typeof value === 'string') {
      params.set(key, value);
    }
  });
  Object.entries(state.multi ?? {}).forEach(([key, values]) => {
    (values ?? []).forEach((value) => params.append(key, value));
  });
  params.set(TRACE_V4_SHARE_URL_PARAM_KEY, viewId);
  return params.toString();
};

export const getTraceV4SavedViewShareUrl = (
  experimentId: string,
  state: CapturedV4ViewState,
  viewId: string,
): string => {
  const route = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces);
  return `${window.location.origin}${window.location.pathname}#${route}?${buildV4ViewQuery(state, viewId)}`;
};

/**
 * Decode the comma-joined `cols` value into known column ids in the given order. Ids that no longer
 * resolve to a known column are dropped (a view saved against an older column set still opens);
 * returns undefined when the value is absent or nothing resolves, so the caller falls back to the
 * user's own columns rather than hiding every column.
 */
export const decodePreviewColumns = (
  raw: string | undefined | null,
  allColumns: readonly TraceColumnId[],
): TraceColumnId[] | undefined => {
  if (!raw) {
    return undefined;
  }
  const known = new Set<string>(allColumns);
  const resolved = raw
    .split(COLUMNS_SEPARATOR)
    .filter(Boolean)
    .filter((id): id is TraceColumnId => known.has(id));
  return resolved.length > 0 ? resolved : undefined;
};
