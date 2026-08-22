import { FormattedMessage } from 'react-intl';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { DetectIssuesButton } from '@databricks/web-shared/genai-traces-table';
import {
  TraceFilterButton,
  type ColumnSelectorGroup,
  type ColumnSelectorOption,
  type SortDirection,
  type TraceColumnId,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { shouldEnableIssueDetection } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { type TracesV4AssessmentColumns } from '../hooks/useTracesV4AssessmentColumns';
import { type TracesV4Density } from '../hooks/useTracesV4Density';
import { TracesV4DateSelector, TracesV4RefreshButton } from './TracesV4DateSelector';
import { TracesV4DisplayButton } from './TracesV4DisplayButton';
import { TracesV4ActionsButton } from './TracesV4ActionsButton';
import { type TracesV4TraceActions } from '../hooks/useTracesV4TraceActions';
import { useMlflowTraceFilterFields } from '../utils/filterModel';

export interface TracesV4ToolbarParams {
  filterModel: TraceFilterModel;
  onFilterChange: (next: TraceFilterModel) => void;
  /** Clear every active filter — popover clauses and URL tag filters both. */
  onClearFilters: () => void;
  activeFilterCount: number;
  visibleColumns: TraceColumnId[];
  onToggleColumn: (column: TraceColumnId) => void;
  onResetColumns: () => void;
  /** Assessment column selection (dynamic, per-page) rendered as a group in the column selector. */
  assessmentColumns: TracesV4AssessmentColumns;
  /** Active sort column + direction, and its setter — surfaced in the Display popover's Sort submenu. */
  sort: TraceColumnId;
  dir: SortDirection;
  onSort: (column: TraceColumnId, direction: SortDirection) => void;
  /** Row-height density + setter — surfaced in the Display popover's Row height submenu. */
  density: TracesV4Density;
  onDensityChange: (density: TracesV4Density) => void;
  selectionCount: number;
  onBulkDelete: () => void;
  /** True while any trace-search query is fetching — drives the refresh button's spin state. */
  isRefreshing: boolean;
  experimentId: string;
  /** Shared trace-action building blocks powering the "Use for evaluation" group. */
  actions: TracesV4TraceActions;
  /** The full cross-page selection — each carries its `ModelTraceInfoV3`. */
  selectedTraceInfos: ModelTraceInfoV3[];
  /** Opens the AI issue-detection modal (seeded with the current selection). Omit to hide the button. */
  onDetectIssues?: () => void;
  /** The "Views" saved-views dropdown, rendered far left (before the date selector). Omit to hide it. */
  savedViewsButton?: React.ReactNode;
}

// Column-selector options: static componentIds (lint requires static ids) + localized labels. Adding
// a column to `TraceColumnId` triggers a TypeScript exhaustiveness check on this map.
const COLUMN_LABELS: Record<TraceColumnId, React.ReactNode> = {
  trace_id: <FormattedMessage defaultMessage="Trace ID" description="Column selector label for the trace-id column" />,
  trace_name: (
    <FormattedMessage defaultMessage="Trace name" description="Column selector label for the trace-name column" />
  ),
  start_time: <FormattedMessage defaultMessage="Time" description="Column selector label for the start-time column" />,
  input: <FormattedMessage defaultMessage="Input" description="Column selector label for the input column" />,
  output: <FormattedMessage defaultMessage="Output" description="Column selector label for the output column" />,
  user: <FormattedMessage defaultMessage="User" description="Column selector label for the user column" />,
  session: <FormattedMessage defaultMessage="Session" description="Column selector label for the session column" />,
  duration: <FormattedMessage defaultMessage="Duration" description="Column selector label for the duration column" />,
  state: <FormattedMessage defaultMessage="State" description="Column selector label for the state column" />,
  source: <FormattedMessage defaultMessage="Source" description="Column selector label for the source column" />,
  run_name: <FormattedMessage defaultMessage="Run name" description="Column selector label for the run-name column" />,
  tokens: <FormattedMessage defaultMessage="Tokens" description="Column selector label for the tokens column" />,
  cost: <FormattedMessage defaultMessage="Cost" description="Column selector label for the cost column" />,
  tags: <FormattedMessage defaultMessage="Tags" description="Column selector label for the tags column" />,
};

// Static per-column componentIds (the lint rule requires static ids). Canonical render order.
const COLUMN_OPTIONS: ColumnSelectorOption[] = [
  { id: 'trace_id', label: COLUMN_LABELS.trace_id, componentId: 'mlflow.traces-v4.column-selector.item.trace_id' },
  {
    id: 'trace_name',
    label: COLUMN_LABELS.trace_name,
    componentId: 'mlflow.traces-v4.column-selector.item.trace_name',
  },
  {
    id: 'start_time',
    label: COLUMN_LABELS.start_time,
    componentId: 'mlflow.traces-v4.column-selector.item.start_time',
  },
  { id: 'input', label: COLUMN_LABELS.input, componentId: 'mlflow.traces-v4.column-selector.item.input' },
  { id: 'output', label: COLUMN_LABELS.output, componentId: 'mlflow.traces-v4.column-selector.item.output' },
  { id: 'user', label: COLUMN_LABELS.user, componentId: 'mlflow.traces-v4.column-selector.item.user' },
  { id: 'session', label: COLUMN_LABELS.session, componentId: 'mlflow.traces-v4.column-selector.item.session' },
  { id: 'duration', label: COLUMN_LABELS.duration, componentId: 'mlflow.traces-v4.column-selector.item.duration' },
  { id: 'state', label: COLUMN_LABELS.state, componentId: 'mlflow.traces-v4.column-selector.item.state' },
  { id: 'source', label: COLUMN_LABELS.source, componentId: 'mlflow.traces-v4.column-selector.item.source' },
  {
    id: 'run_name',
    label: COLUMN_LABELS.run_name,
    componentId: 'mlflow.traces-v4.column-selector.item.run_name',
  },
  { id: 'tokens', label: COLUMN_LABELS.tokens, componentId: 'mlflow.traces-v4.column-selector.item.tokens' },
  { id: 'cost', label: COLUMN_LABELS.cost, componentId: 'mlflow.traces-v4.column-selector.item.cost' },
  { id: 'tags', label: COLUMN_LABELS.tags, componentId: 'mlflow.traces-v4.column-selector.item.tags' },
];

export interface TracesV4ToolbarSlots {
  /** Rendered before the shared search box: the Views dropdown, then the date-range selector. */
  leftControls: React.ReactNode;
  /** Rendered after the shared search box: filters, columns, a spacer, Detect Issues, and refresh. */
  rightControls: React.ReactNode;
}

/**
 * Builds the V4-specific controls for the shared `TracesTableView`. Filters and columns stay beside
 * the shared search while a flexible spacer pins refresh and Analyze right. None depend on trace
 * data, so they paint immediately.
 */
export const useTracesV4ToolbarSlots = ({
  filterModel,
  onFilterChange,
  onClearFilters,
  activeFilterCount,
  visibleColumns,
  onToggleColumn,
  onResetColumns,
  assessmentColumns,
  sort,
  dir,
  onSort,
  density,
  onDensityChange,
  selectionCount,
  onBulkDelete,
  isRefreshing,
  experimentId,
  actions,
  selectedTraceInfos,
  onDetectIssues,
  savedViewsButton,
}: TracesV4ToolbarParams): TracesV4ToolbarSlots => {
  const hasSelection = selectionCount > 0;
  const filterFields = useMlflowTraceFilterFields(assessmentColumns.candidateNames);

  // Assessment columns are dynamic (per-page), so they render as a labeled group under the standard
  // columns in the Display → Columns submenu. Omitted entirely when the page has no assessments.
  const columnGroups: ColumnSelectorGroup[] | undefined =
    assessmentColumns.selectorOptions.length > 0
      ? [
          {
            label: (
              <FormattedMessage
                defaultMessage="Assessments"
                description="Section label for assessment columns in the traces table column selector"
              />
            ),
            options: assessmentColumns.selectorOptions,
            visibleIds: assessmentColumns.visibleIds,
            onToggle: assessmentColumns.toggle,
          },
        ]
      : undefined;

  // Note: the Databricks build disables Delete for UC-backed traces (with a "delete from the Delta
  // table instead" tooltip). OSS traces are always deletable via the standard path, so that gate is
  // dropped here and Delete is always enabled.
  return {
    leftControls: (
      <>
        {savedViewsButton}
        <TracesV4DateSelector experimentId={experimentId} />
      </>
    ),
    rightControls: (
      <>
        <TraceFilterButton
          fields={filterFields}
          filterModel={filterModel}
          onChange={onFilterChange}
          onClearAll={onClearFilters}
          activeCount={activeFilterCount}
        />
        <TracesV4DisplayButton
          columns={COLUMN_OPTIONS}
          visibleColumns={visibleColumns}
          onToggleColumn={onToggleColumn}
          onResetColumns={onResetColumns}
          columnGroups={columnGroups}
          sortColumnLabels={COLUMN_LABELS}
          sort={sort}
          dir={dir}
          onSort={onSort}
          density={density}
          onDensityChange={onDensityChange}
        />
        {hasSelection && (
          <TracesV4ActionsButton
            selectionCount={selectionCount}
            onDelete={onBulkDelete}
            experimentId={experimentId}
            actions={actions}
            selectedTraceInfos={selectedTraceInfos}
          />
        )}
        <div css={{ flex: 1 }} />
        {shouldEnableIssueDetection() && onDetectIssues && (
          <DetectIssuesButton componentId="mlflow.traces-v4.detect-issues-button" onClick={onDetectIssues} />
        )}
        <TracesV4RefreshButton isFetching={isRefreshing} />
      </>
    ),
  };
};
