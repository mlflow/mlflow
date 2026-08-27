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
 * The translation is intentionally lossy in the ways that don't matter to the user: it maps the parts
 * a saved view is actually judged by (columns, sort, time range) and best-effort maps filters, while
 * dropping V3-internal fields with no V4 equivalent.
 */

// The prefix V3 wrote its saved-view tags under. Kept in sync with `TRACE_SAVED_VIEW_TAG_PREFIX` in
// `traces-v3/TracesV3SavedViews.tsx` (which stays frozen); duplicated here so the V4 tab can read
// legacy tags without importing the V3 module.
export const TRACE_V3_SAVED_VIEW_TAG_PREFIX = 'mlflow.traceViewState.';

const V3_SORT_SEPARATOR = '::';

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
 * columns are a comma-joined id list, and `filter` is repeatable. Every field is optional — a view
 * may omit any of them — and unknown extra fields are ignored.
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

/**
 * Split V3's monolithic `key::type::asc` sort string into V4's separate `sort` + `dir`. The middle
 * `type` segment is dropped: V4 derives a column's sort type from its own column definition, so the
 * stored type is redundant rather than lost. Returns an empty object for an absent or malformed
 * value so a view with a broken sort still opens (just unsorted).
 */
const translateV3Sort = (sort: string | undefined): Pick<CapturedV4ViewState['single'], 'sort' | 'dir'> => {
  if (!sort) {
    return {};
  }
  const parts = sort.split(V3_SORT_SEPARATOR);
  if (parts.length !== 3) {
    return {};
  }
  const [key, , ascStr] = parts;
  if (!key) {
    return {};
  }
  return { sort: key, dir: ascStr === 'true' ? 'asc' : 'desc' };
};

/**
 * Best-effort map V3's generic `filter[]` onto V4's `tag[]`. V4 tags are strictly `key=value`
 * trace-metadata filters, whereas V3 filters were a looser vocabulary, so only entries already
 * shaped `key=value` carry over; anything else is dropped rather than producing an invalid V4 tag.
 * (In OSS with URL persistence off, V3 filters live in localStorage and usually aren't captured into
 * a view at all, so this is rarely exercised.) Returns undefined when nothing maps.
 */
const translateV3Filters = (filters: string[] | undefined): string[] | undefined => {
  if (!filters || filters.length === 0) {
    return undefined;
  }
  const mapped = filters.filter((entry) => {
    const eq = entry.indexOf('=');
    // Require a non-empty key and value on either side of the first '='.
    return eq > 0 && eq < entry.length - 1;
  });
  return mapped.length > 0 ? mapped : undefined;
};

/**
 * Translate a deserialized legacy V3 saved-view state into V4's captured view state, so a V3-authored
 * view can be applied by the V4 tab's normal navigation path. Maps columns (`selectedColumns` →
 * `cols`), sort (`key::type::asc` → `sort`+`dir`), the time range (identical keys, passthrough), and
 * best-effort filters (`filter[]` → `tag[]`). V3-internal fields (`viewState`) are dropped. The
 * caller is responsible for having already inflated the envelope's `state` blob.
 *
 * The V4 captured state's `filters` (the popover filter model) is intentionally left absent: V3's
 * `filter[]` are `key=value` trace-metadata filters that map onto V4's URL-backed `tag[]` above, not
 * onto the in-memory popover model, which V3 had no equivalent of. An absent `filters` reads as an
 * empty model, so a translated V3 view opens with no popover clauses and isn't spuriously dirty.
 */
export const translateV3ViewState = (state: V3SavedViewState): CapturedV4ViewState => {
  const v3Single = state.single ?? {};
  const single: CapturedV4ViewState['single'] = {};

  if (v3Single.selectedColumns !== undefined) {
    single.cols = v3Single.selectedColumns;
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

  const multi: CapturedV4ViewState['multi'] = {};
  const tag = translateV3Filters(state.multi?.filter);
  if (tag !== undefined) {
    multi.tag = tag;
  }

  return { single, multi };
};
