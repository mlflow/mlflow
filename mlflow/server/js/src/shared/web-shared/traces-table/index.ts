// Public barrel for `@databricks/web-shared/traces-table`: a dumb, fully-controlled presentational
// traces table + controls that owns no data, no URL state, and no product coupling. The `data/`
// hooks are an opt-in fetch layer the presentational components never import.

// Types
export type { TraceColumnId, SortDirection, PageSize, TraceTableColumn, SessionHrefGetter } from './types';

// Constants
export {
  TRACE_COLUMN_IDS,
  SORTABLE_TRACE_COLUMNS,
  isSortableTraceColumn,
  DEFAULT_SORT_COLUMN,
  DEFAULT_SORT_DIR,
  PAGE_SIZE_OPTIONS,
  DEFAULT_PAGE_SIZE,
  COLUMN_SIZES,
  type ColumnSizeSpec,
} from './constants';

// Columns
export { STANDARD_COLUMNS, getVisibleColumnDefs, getTableMeta, openLabel, type TracesTableMeta } from './columns';

// Presentational components
export { TracesTable, type TracesTableProps } from './TracesTable';
export { TracesTableToolbar, type TracesTableToolbarProps } from './TracesTableToolbar';
export { TracesPaginationBar, type TracesPaginationBarProps } from './TracesPaginationBar';
export {
  TraceColumnSelector,
  type TraceColumnSelectorProps,
  type ColumnSelectorOption,
  type GenericColumnOption,
  type ColumnSelectorGroup,
} from './TraceColumnSelector';
export { TraceFilterButton, type TraceFilterButtonProps } from './TraceFilterButton';
export { TracesTableView, type TracesTableViewProps, type TracesTableViewState } from './TracesTableView';

// State components
export {
  TracesEmptyState,
  TracesNoResultsState,
  type TracesNoResultsStateProps,
  TracesNoMoreResultsState,
  type TracesNoMoreResultsStateProps,
  TracesErrorState,
  type TracesErrorStateProps,
  TracesErrorAlert,
  type TracesErrorAlertProps,
  type ErrorDescriptionGetter,
} from './TracesTableStates';

// Cell renderers (exported for consumers building custom columns / tests)
export {
  TraceIdCell,
  TraceNameCell,
  TraceInputCell,
  TraceOutputCell,
  TraceUserCell,
  TraceSessionCell,
  TraceStateCell,
  TraceSourceCell,
  TraceRunNameCell,
  TraceStartTimeCell,
  TraceDurationCell,
  TraceTokensCell,
  TraceCostCell,
  TraceTagsCell,
} from './TraceCell';

// Filter model (generic AST + UI helpers; API-specific compilation stays consumer-side)
export {
  FilterOp,
  EMPTY_FILTER_MODEL,
  isClauseComplete,
  countActiveFilters,
  makeEmptyClause,
  type FilterClause,
  type FilterFieldDef,
  type FilterFieldSelectOption,
  type FilterValueInputKind,
  type TraceFilterModel,
} from './filterModel';

// Helpers
export { formatTraceDuration } from './formatTraceDuration';

// Hooks (presentational state)
export { useBulkTraceSelection, type UseBulkTraceSelectionResult } from './hooks/useBulkTraceSelection';
export {
  useTraceColumnVisibility,
  type UseTraceColumnVisibilityParams,
  type TraceColumnVisibility,
} from './hooks/useTraceColumnVisibility';
export {
  useTraceColumnSizing,
  type UseTraceColumnSizingParams,
  type TraceColumnSizing,
} from './hooks/useTraceColumnSizing';

// Opt-in data layer — the reusable cursor-paginated fetch mechanism. Presentational components never
// import from here; a consumer with a different backend ignores these and feeds the table itself.
export {
  useTracesPageQuery,
  buildSearchTracesPagePayload,
  type UseTracesPageQueryParams,
  type TracesPageQueryResult,
  type TracesQueryIdentity,
  type SearchTracesPageResponse,
  type SearchTracesPagePayload,
} from './data/useTracesPageQuery';
export { useTraceTokenCache, PageTokenStack, type TraceTokenCache } from './data/useTraceTokenCache';
export { fetchTracesLongRunningPage } from './data/searchTracesLongRunningPage';
export { fetchTracesProgressivePage } from './data/searchTracesProgressivePage';
