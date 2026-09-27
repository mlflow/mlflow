import {
  FilterOp,
  isSortableTraceColumn,
  type FilterClause,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { type CapturedV4ViewState } from './tracesV4SavedViewState';

/**
 * Compatibility shim for legacy V3 saved views.
 *
 * V3 is being removed, so nothing writes the V3 format anymore — but views authored while V3 shipped
 * are stored as experiment tags under the V3 prefix and must stay usable after V4 replaces V3. This
 * module lets the V4 tab DISCOVER those tags (via {@link TRACE_V3_SAVED_VIEW_TAG_PREFIX} /
 * {@link getTraceV3SavedViewTagKey}) and READ them (via {@link translateV3ViewState}), translating
 * the frozen V3 state shape into V4's captured view state. It never WRITES the V3 format: a legacy
 * view opens and can be deleted in place, and overwriting one migrates it forward — the edited state
 * is written under the V4 prefix (same id) and the old V3 tag is deleted (see `overwriteView`).
 *
 * V3 and V4 speak different vocabularies for the three things a saved view is judged by — columns,
 * sort, and filters — so the translation maps each across:
 *   - columns: V3 ids (`request`, `response`, `request_time`, `execution_duration`, …) → V4 ids
 *     (`input`, `output`, `start_time`, `duration`, …); ids with no V4 column are dropped.
 *   - sort: V3's monolithic `key::type::asc` → V4's `sort`+`dir`, with the key mapped and dropped
 *     entirely if it isn't a V4-sortable column.
 *   - filters: V3's URL-shaped `column::operator::value::key` entries → V4's popover filter model
 *     ({@link FilterClause}[]); columns/operators with no V4 equivalent are dropped.
 * The mapping is intentionally lossy in the ways that don't matter: an unmappable column/sort/filter
 * is dropped rather than carried as an invalid V4 value.
 */

// The prefix V3 wrote its saved-view tags under. Kept in sync with `TRACE_SAVED_VIEW_TAG_PREFIX` in
// `traces-v3/TracesV3SavedViews.tsx` (which stays frozen); duplicated here so the V4 tab can read
// legacy tags without importing the V3 module.
export const TRACE_V3_SAVED_VIEW_TAG_PREFIX = 'mlflow.traceViewState.';

// V3 joins both its `sort` string (`key::type::asc`) and each `filter` entry
// (`column::operator::value::key`) on the same separator (see `useFilters.tsx` / `TracesV3SavedViews`).
const V3_VALUE_SEPARATOR = '::';

export const getTraceV3SavedViewTagKey = (id: string): string => `${TRACE_V3_SAVED_VIEW_TAG_PREFIX}${id}`;

export const getTraceV3SavedViewIdFromTagKey = (key: string): string | null => {
  if (!key.startsWith(TRACE_V3_SAVED_VIEW_TAG_PREFIX)) {
    return null;
  }
  // A key that is exactly the prefix (no id) yields an empty id, which would collide across tags;
  // treat it as not a saved-view key (mirrors the V3 / V4 helpers).
  const id = key.slice(TRACE_V3_SAVED_VIEW_TAG_PREFIX.length);
  return id === '' ? null : id;
};

/**
 * The frozen shape of a deserialized V3 saved-view state. Mirrors V3's `CapturedTraceViewState`
 * (`traces-v3/TracesV3SavedViews.tsx`): `sort` is the monolithic `key::type::asc` wire string,
 * columns are a comma-joined id list, and `filter` is repeatable (each `column::operator::value::key`).
 * Every field is optional — a view may omit any of them — and unknown extra fields are ignored.
 */
export interface V3SavedViewState {
  single?: {
    selectedColumns?: string;
    sort?: string;
    viewState?: string;
    startTimeLabel?: string;
    startTime?: string;
    endTime?: string;
  };
  multi?: {
    filter?: string[];
  };
}

// V3 → V4 column-id map. V3 named a handful of columns differently (`request`/`response`/
// `request_time`/`execution_duration`); the rest share the same id in both. Ids absent here (e.g.
// V3's `logged_model`, `custom_metadata.*`, `span.*`) have no V4 column and are dropped.
const V3_TO_V4_COLUMN_ID = new Map<string, string>([
  ['request', 'input'],
  ['response', 'output'],
  ['request_time', 'start_time'],
  ['execution_duration', 'duration'],
  ['trace_id', 'trace_id'],
  ['trace_name', 'trace_name'],
  ['user', 'user'],
  ['session', 'session'],
  ['state', 'state'],
  ['source', 'source'],
  ['run_name', 'run_name'],
  ['tokens', 'tokens'],
  ['tags', 'tags'],
]);

// V3 filter `column` → V4 popover filter field id. Standard scalar columns map straight across;
// V3's TAG / ASSESSMENT groups become V4's key-requiring `tag` / `assessment` fields (the V3 entry's
// 4th segment carries the key). Columns with no V4 filter field (logged_model, prompt, git_*,
// custom_metadata.*, expectation, issue.id, span.content) are dropped.
const V3_TO_V4_FILTER_FIELD = new Map<string, string>([
  ['execution_duration', 'duration'],
  ['state', 'state'],
  ['user', 'user'],
  ['session', 'session'],
  ['run_name', 'run_name'],
  ['trace_name', 'trace_name'],
  ['source', 'source'],
  ['request', 'input'],
  ['response', 'output'],
  ['span.name', 'span_name'],
  ['span.type', 'span_type'],
  ['span.status', 'span_status'],
  ['TAG', 'tag'],
  ['ASSESSMENT', 'assessment'],
]);

// V4 popover fields that carry a free-text key alongside their value (mirrors `requiresKey` in
// `filterModel.ts`): a translated clause for one of these must keep the V3 entry's key segment.
const V4_KEY_REQUIRING_FIELDS = new Set<string>(['tag', 'metadata', 'assessment']);

// The operator tokens V4 understands (the `FilterOp` enum values). V3 additionally had
// `IS NULL` / `IS NOT NULL`, which V4 has no equivalent for — clauses using them are dropped.
const V4_FILTER_OPS = new Set<string>(Object.values(FilterOp));

// V3 → V4 sort-key map: only the two server-sortable V3 columns have a V4-sortable counterpart.
const V3_TO_V4_SORT_KEY = new Map<string, string>([
  ['execution_duration', 'duration'],
  ['request_time', 'start_time'],
]);

/**
 * Map V3's comma-joined `selectedColumns` id list onto V4 column ids, dropping any id with no V4
 * column. Returns the comma-joined V4 list, or undefined when nothing maps (so the caller leaves the
 * user's columns untouched rather than hiding everything). The caller still re-validates against the
 * live column set, so this only needs to translate the names it knows.
 */
const translateV3Columns = (selectedColumns: string | undefined): string | undefined => {
  if (selectedColumns === undefined) {
    return undefined;
  }
  const mapped = selectedColumns
    .split(',')
    .filter(Boolean)
    .map((id) => V3_TO_V4_COLUMN_ID.get(id))
    .filter((id): id is string => id !== undefined);
  return mapped.length > 0 ? mapped.join(',') : undefined;
};

/**
 * Split V3's monolithic `key::type::asc` sort string into V4's separate `sort` + `dir`, mapping the
 * V3 sort key to its V4 column id. The middle `type` segment is dropped (V4 derives a column's sort
 * type from its own definition). Returns an empty object for an absent / malformed value, or for a
 * key that isn't a V4-sortable column, so the view opens (just unsorted) rather than carrying a sort
 * V4 would silently reject.
 */
const translateV3Sort = (sort: string | undefined): Pick<CapturedV4ViewState['single'], 'sort' | 'dir'> => {
  if (!sort) {
    return {};
  }
  const parts = sort.split(V3_VALUE_SEPARATOR);
  if (parts.length !== 3) {
    return {};
  }
  const [v3Key, , ascStr] = parts;
  const v4Key = V3_TO_V4_SORT_KEY.get(v3Key);
  if (!v4Key || !isSortableTraceColumn(v4Key)) {
    return {};
  }
  return { sort: v4Key, dir: ascStr === 'true' ? 'asc' : 'desc' };
};

/**
 * Translate V3's URL-shaped `column::operator::value::key` filter entries into V4's popover filter
 * model. Each entry is mapped by column → V4 field id and validated for a known operator; entries
 * whose column has no V4 field, or whose operator V4 doesn't support (e.g. `IS NULL`), are dropped.
 * A key-requiring field (tag / assessment) keeps the entry's key segment; others drop it. Returns
 * undefined when nothing maps, so a filter-less view stays byte-identical to a native V4 one (and the
 * downstream `isSupportedFilterClause` pass still prunes any clause a field's own operator set rejects).
 */
const translateV3Filters = (filters: string[] | undefined): TraceFilterModel | undefined => {
  if (!filters || filters.length === 0) {
    return undefined;
  }
  const clauses: FilterClause[] = [];
  for (const entry of filters) {
    const [column, operator, value, key] = entry.split(V3_VALUE_SEPARATOR);
    if (!column || !operator) {
      continue;
    }
    const field = V3_TO_V4_FILTER_FIELD.get(column);
    if (!field || !V4_FILTER_OPS.has(operator)) {
      continue;
    }
    const requiresKey = V4_KEY_REQUIRING_FIELDS.has(field);
    // A key-requiring field with no key can't produce a valid clause; a value is always required.
    if ((requiresKey && !key) || !value) {
      continue;
    }
    clauses.push({
      field,
      operator: operator as FilterOp,
      value,
      ...(requiresKey ? { key } : {}),
    });
  }
  return clauses.length > 0 ? clauses : undefined;
};

/**
 * Translate a deserialized legacy V3 saved-view state into V4's captured view state, so a V3-authored
 * view can be applied by the V4 tab's normal navigation path. Maps columns (`selectedColumns` →
 * `cols`, V3→V4 ids), sort (`key::type::asc` → `sort`+`dir`, key mapped), the time range (identical
 * keys, passthrough), and filters (`filter[]` → V4's popover `filters` model). V3-internal fields
 * (`viewState`) are dropped. The caller is responsible for having already inflated the envelope's
 * `state` blob.
 */
export const translateV3ViewState = (state: V3SavedViewState): CapturedV4ViewState => {
  const v3Single = state.single ?? {};
  const single: CapturedV4ViewState['single'] = {};

  const cols = translateV3Columns(v3Single.selectedColumns);
  if (cols !== undefined) {
    single.cols = cols;
  }
  Object.assign(single, translateV3Sort(v3Single.sort));
  // Time-range fields share the same keys and semantics between V3 and V4 — pass them through as-is.
  if (v3Single.startTimeLabel !== undefined) {
    single.startTimeLabel = v3Single.startTimeLabel;
  }
  if (v3Single.startTime !== undefined) {
    single.startTime = v3Single.startTime;
  }
  if (v3Single.endTime !== undefined) {
    single.endTime = v3Single.endTime;
  }

  const result: CapturedV4ViewState = { single, multi: {} };
  const filters = translateV3Filters(state.multi?.filter);
  if (filters !== undefined) {
    result.filters = filters;
  }
  return result;
};
