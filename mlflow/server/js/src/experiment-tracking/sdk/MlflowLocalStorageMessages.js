/**
 * This class contains definitions of message entities corresponding to data stored in LocalStorage.
 * The backwards-compatibility behavior of these messages is as follows:
 *
 * Backwards-compatible changes:
 * 1) Adding a new field: Backwards-compatible. New fields that are absent from old data in
 *    local storage will take on the specified default value.
 * 2) Removing a field: Backwards-compatible. Unknown fields from old data in local storage will be
 *    ignored at construction-time.
 *
 * Backwards-incompatible changes (AVOID MAKING SUCH CHANGES):
 * 1) Changing the type of a field. Old data loaded from local storage will be of the wrong type.
 * 2) Changing the role/usage of a field. It's better to add a new field than to repurpose an
 *    existing field, since a repurposed field may be populated with unexpected data cached in
 *    local storage.
 */
import Immutable from 'immutable';
import { ColumnTypes } from '../constants';

/**
 * This class wraps attributes of the ExperimentPage component's state that should be
 * persisted in / restored from local storage.
 */
export const ExperimentPagePersistedState = Immutable.Record(
  {
    // Comma-separated string containing containing the keys of parameters to display
    paramKeyFilterString: '',
    // Comma-separated string containing containing the keys of metrics to display
    metricKeyFilterString: '',
    // SQL-like query string used to filter runs, e.g. "params.alpha = '0.5'"
    searchInput: '',
    // Canonical order_by key like "params.`alpha`". May be null to indicate the table
    // should use the natural row ordering provided by the server.
    orderByKey: null,
    // Whether the order imposed by orderByKey should be ascending or descending.
    orderByAsc: false,
    // Filter key to show results based on start time
    startTime: null,
  },
  'ExperimentPagePersistedState',
);
/**
 * This class wraps attributes of the ExperimentPage component's state that should be
 * persisted in / restored from local storage.
 */
export const ExperimentViewPersistedState = Immutable.Record(
  {
    // Object mapping run UUIDs (strings) to booleans, where a boolean value of true indicates that
    // a run has been minimized (its child runs are hidden).
    runsHiddenByExpander: {},
    // Object mapping run UUIDs (strings) to booleans, where a boolean value of true indicates that
    // a run has been expanded (its child runs are visible).
    runsExpanded: {},
    // If true, shows the multi-column table view instead of the compact table view.
    showMultiColumns: true,
    // Arrays of "unbagged", or split-out metric and param keys (strings). We maintain these as
    // lists to help keep them ordered (i.e. splitting out a column shouldn't change the ordering of
    // columns that have already been split out)
    unbaggedMetrics: [],
    unbaggedParams: [],
    // Unchecked keys in the columns dropdown
    categorizedUncheckedKeys: {
      [ColumnTypes.ATTRIBUTES]: [],
      [ColumnTypes.PARAMS]: [],
      [ColumnTypes.METRICS]: [],
      [ColumnTypes.TAGS]: [],
    },
  },
  'ExperimentViewPersistedState',
);

/**
 * This class wraps persisted states for AgGrid based table.
 */
export const AgGridPersistedState = Immutable.Record({
  // column group open/close state
  columnGroupState: [],
});
