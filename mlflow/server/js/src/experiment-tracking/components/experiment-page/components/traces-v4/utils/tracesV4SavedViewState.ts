import { type TraceColumnId, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';

/**
 * Pure serialization for V4 saved views.
 *
 * Most V4 view state is URL-first: search, sort, page size, tag filters and the time range all live
 * in the URL, so a view is largely a snapshot of the URL search string (minus the transient
 * page/traceId/share-key params). The one piece that ISN'T in the URL — column visibility — is
 * stored in the envelope under the `cols` key and, on open, restored into the user's column store
 * (localStorage) rather than the URL. Applying a view is a navigation to the rebuilt query plus that
 * column restore; the live table then diverges from the stored view as the user edits ("dirty"),
 * and Overwrite / Reset act on the active view.
 */

// Distinct from the V3 prefix (`mlflow.traceViewState.`) and the runs prefix
// (`mlflow.sharedViewState.`): V3's `sort` wire format (`key::type::asc`) is incompatible with V4's
// separate `sort`+`dir` params, so a V4 list must never show a V3 or runs view (and vice-versa).
export const TRACE_V4_SAVED_VIEW_TAG_PREFIX = 'mlflow.tracesV4ViewState.';
export const TRACE_V4_SHARE_URL_PARAM_KEY = 'traceViewShareKey';

// The key under which column visibility is stored inside the envelope's captured state. It is NOT a
// URL param: columns are restored into the user's column store on open, not carried in the query.
export const TRACE_V4_COLS_PARAM_KEY = 'cols';

const COLUMNS_SEPARATOR = ',';

// The URL view params that make up a V4 saved view. `tag` is repeatable (one param per filter).
// `groupBy` carries the session-grouping toggle (`groupBy=session`). Deliberately EXCLUDES the
// transient params `page`, `traceId`, and the share key itself.
const SINGLE_VALUE_KEYS = [
  'q',
  'pageSize',
  'sort',
  'dir',
  'startTimeLabel',
  'startTime',
  'endTime',
  'groupBy',
] as const;
const MULTI_VALUE_KEYS = ['tag'] as const;

export interface CapturedV4ViewState {
  single: Partial<Record<(typeof SINGLE_VALUE_KEYS)[number] | typeof TRACE_V4_COLS_PARAM_KEY, string>>;
  multi: Partial<Record<(typeof MULTI_VALUE_KEYS)[number], string[]>>;
  // The popover filter clauses (React state, not URL-backed), captured so a saved view restores the
  // exact filter model the user had applied. Absent in older stored views; restored through a
  // validation pass (see `isSupportedFilterClause`) so a clause referencing a since-removed
  // field/operator is dropped rather than silently producing wrong results.
  filters?: TraceFilterModel;
  // Assessment-column visibility by assessment name (localStorage-backed, not URL). A full map, not a
  // visible-id list, so a hidden column is recorded too — restoring a bare id list would lose explicit
  // hides. Absent in older stored views (treated as "capture nothing", i.e. clear overrides on restore).
  assessmentColumns?: Record<string, boolean>;
  /** Custom tag/metadata column visibility. Optional for backward compatibility with older views. */
  customColumns?: Record<string, boolean>;
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
  MULTI_VALUE_KEYS.some((key) => params.getAll(key).length > 0);

/**
 * Capture the current view: the whitelisted URL params, the live visible columns (which live in
 * localStorage, not the URL, so they're passed in rather than read from `params`), and the live
 * popover filter model (also React state, not URL-backed). The incoming URL's own `cols` / share
 * key are intentionally ignored so opening view A then saving view B never leaks A's columns or id
 * into B. An empty filter model is omitted so a filter-less view stays byte-identical to a legacy
 * one (and never spuriously reads as dirty against `filters ?? []`).
 */
export const captureV4ViewState = (
  params: URLSearchParams,
  visibleColumns: readonly TraceColumnId[],
  filterModel: TraceFilterModel = [],
  assessmentColumns: Record<string, boolean> = {},
  customColumns: Record<string, boolean> = {},
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
  const state: CapturedV4ViewState = { single, multi };
  if (filterModel.length > 0) {
    state.filters = filterModel;
  }
  // Omit an empty map so a view saved on a page with no assessments stays byte-identical to a legacy
  // one (and never spuriously reads as dirty against `assessmentColumns ?? {}`).
  if (Object.keys(assessmentColumns).length > 0) {
    state.assessmentColumns = assessmentColumns;
  }
  // Omit an empty map so a view saved on a page with no custom columns stays byte-identical to a legacy
  // one (and never spuriously reads as dirty against `customColumns ?? {}`).
  if (Object.keys(customColumns).length > 0) {
    state.customColumns = customColumns;
  }
  return state;
};

/**
 * Rebuild a URL query string from a captured view + the view's id (as the share key). Columns are
 * intentionally excluded — they live in the envelope's `cols` key and are restored into the user's
 * column store on open, not carried in the URL (decode them with {@link decodeViewColumns}).
 */
export const buildV4ViewQuery = (state: CapturedV4ViewState, viewId: string): string => {
  const params = new URLSearchParams();
  Object.entries(state.single ?? {}).forEach(([key, value]) => {
    if (typeof value === 'string' && key !== TRACE_V4_COLS_PARAM_KEY) {
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
 * Decode a view's stored column set (the envelope's comma-joined `cols` value) into known column ids
 * in stored order. Ids that no longer resolve to a known column are dropped (a view saved against an
 * older column set still opens); returns undefined when the value is absent or nothing resolves, so
 * the caller can leave the user's columns untouched rather than hiding every column.
 */
export const decodeViewColumns = (
  state: CapturedV4ViewState,
  allColumns: readonly TraceColumnId[],
): TraceColumnId[] | undefined => {
  const raw = state.single?.[TRACE_V4_COLS_PARAM_KEY];
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
