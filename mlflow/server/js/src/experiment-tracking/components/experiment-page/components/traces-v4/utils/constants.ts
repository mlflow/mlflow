/** Search-input debounce. Matches datasets-v2 — snappy feedback without wasted backend calls. */
export const SEARCH_DEBOUNCE_MS = 250;

/** localStorage namespace for the traces-v4 column-visibility preference. */
export const TRACE_COLUMN_STORAGE_KEY_PREFIX = 'mlflow.traces-v4.columns';

/** localStorage namespace for the traces-v4 per-column pixel widths (resizable columns). */
export const TRACE_COLUMN_SIZES_STORAGE_KEY_PREFIX = 'mlflow.traces-v4.column-sizes';

/** localStorage namespace for the traces-v4 assessment-column opt-in/opt-out preference. */
export const TRACE_ASSESSMENT_COLUMN_STORAGE_KEY_PREFIX = 'mlflow.traces-v4.assessment-columns';

/** localStorage namespace for the traces-v4 row-height (density) preference. */
export const TRACE_DENSITY_STORAGE_KEY_PREFIX = 'mlflow.traces-v4.density';
