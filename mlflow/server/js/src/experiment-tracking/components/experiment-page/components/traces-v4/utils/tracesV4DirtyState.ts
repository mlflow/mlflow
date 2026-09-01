import { isEqual } from 'lodash';

import { type CapturedV4ViewState } from './tracesV4SavedViewState';
import { DEFAULT_TRACES_V4_TIME_LABEL } from './timeRange';

/**
 * Dirty comparison for V4 saved views: does the live table state still match the view the user
 * opened? Drives the Views button's dirty dot and the Overwrite / Reset actions.
 *
 * It compares exactly what a view captures — the whitelisted URL params ({@link CapturedV4ViewState}
 * `single`/`multi`), the stored column set, and the popover filter model — and nothing else.
 *
 * The match is narrower than a raw field equality, for the same reasons managed's is:
 *   - Relative time ranges recompute their absolute `startTime`/`endTime` on every render, so for a
 *     non-CUSTOM label only the label is compared (the bounds are dropped). The default label is
 *     explicit in fresh captures but absent in older stored views, so it is normalized in first.
 *   - Column visibility is a selection, not an ordering, so the two lists are compared as sets.
 *   - The popover filter model is deep-compared (`filters ?? []`) so absent (legacy) and empty read
 *     as equal. The caller must normalize the stored baseline through the same `supportedFilters`
 *     validation openView applies, so a clause referencing a since-removed field/operator (dropped
 *     on restore) can't strand the view permanently dirty.
 */

const START_TIME_LABEL_KEY = 'startTimeLabel';

// The `cols` param no longer rides in the URL query, but a stored view may still carry it inside
// `single`; strip it from the query comparison and diff columns as a set instead (below).
const COLS_KEY = 'cols';

/**
 * Canonicalize a captured view's `single`/`multi` params into a stable, comparable query string:
 * drop the recomputed absolute bounds for relative time ranges, normalize the default label, and
 * sort by key so param order never spuriously reads as dirty.
 */
const canonicalViewQuery = (state: CapturedV4ViewState): string => {
  const params = new URLSearchParams();
  Object.entries(state.single ?? {}).forEach(([key, value]) => {
    if (typeof value === 'string' && key !== COLS_KEY) {
      params.set(key, value);
    }
  });
  Object.entries(state.multi ?? {}).forEach(([key, values]) => {
    (values ?? []).forEach((value) => params.append(key, value));
  });

  const label = params.get(START_TIME_LABEL_KEY) ?? DEFAULT_TRACES_V4_TIME_LABEL;
  if (label !== 'CUSTOM') {
    params.delete('startTime');
    params.delete('endTime');
  }
  params.set(START_TIME_LABEL_KEY, label);

  // Sort by key; Array.sort is stable, so repeated keys (tag filters) keep their relative order.
  const entries = [...params.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  const canonical = new URLSearchParams();
  for (const [key, value] of entries) {
    canonical.append(key, value);
  }
  return canonical.toString();
};

/** Column visibility is a selection, not an ordering — compare the two id lists as sets. */
const columnSetsEqual = (a: readonly string[], b: readonly string[]): boolean => {
  if (a.length !== b.length) {
    return false;
  }
  const set = new Set(a);
  return b.every((id) => set.has(id));
};

const colsOf = (state: CapturedV4ViewState): string[] => {
  const raw = state.single?.[COLS_KEY];
  return raw ? raw.split(',').filter(Boolean) : [];
};

/**
 * Compare EFFECTIVE assessment visibility, not the key sets. Visibility is page-derived, so captures
 * on different pages carry different names; an absent name means default-visible, so an extra
 * default-visible entry isn't a change — only a real hide/show reads as dirty. (Two captures across
 * pages with different assessments may still diverge, but the dirty dot is advisory and both
 * Overwrite / Reset recover.)
 */
const assessmentVisibilityEqual = (a: Record<string, boolean> = {}, b: Record<string, boolean> = {}): boolean => {
  const names = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const name of names) {
    if ((a[name] ?? true) !== (b[name] ?? true)) {
      return false;
    }
  }
  return true;
};

/**
 * Compare EFFECTIVE custom column visibility. Unlike assessments (opt-out / default-visible), custom
 * columns are opt-in / default-hidden, so an absent id means hidden. Only a real show/hide reads
 * as dirty — an extra default-hidden entry isn't a change.
 */
const customVisibilityEqual = (a: Record<string, boolean> = {}, b: Record<string, boolean> = {}): boolean => {
  const ids = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const id of ids) {
    if ((a[id] ?? false) !== (b[id] ?? false)) {
      return false;
    }
  }
  return true;
};

/** True when the live captured state matches the stored view (i.e. not dirty). */
export const capturedV4StatesMatch = (live: CapturedV4ViewState, stored: CapturedV4ViewState): boolean =>
  canonicalViewQuery(live) === canonicalViewQuery(stored) &&
  columnSetsEqual(colsOf(live), colsOf(stored)) &&
  assessmentVisibilityEqual(live.assessmentColumns, stored.assessmentColumns) &&
  customVisibilityEqual(live.customColumns, stored.customColumns) &&
  isEqual(live.filters ?? [], stored.filters ?? []);

// Exported for unit-testing the pure comparisons in isolation.
export const __test__ = { canonicalViewQuery, columnSetsEqual, assessmentVisibilityEqual, customVisibilityEqual };
