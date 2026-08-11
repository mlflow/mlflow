import { FormattedMessage, useIntl } from 'react-intl';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { DetectIssuesButton } from '@databricks/web-shared/genai-traces-table';
import {
  TraceColumnSelector,
  TraceFilterButton,
  type ColumnSelectorOption,
  type TraceColumnId,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { shouldEnableIssueDetection } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { type TracesV4AssessmentColumns } from '../hooks/useTracesV4AssessmentColumns';
import { TracesV4DateSelector, TracesV4RefreshButton } from './TracesV4DateSelector';
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
  selectionCount: number;
  onBulkDelete: () => void;
  /** Grays out the Actions → Delete item (UC-backed traces can't be deleted here). */
  isDeleteDisabled: boolean;
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
  start_time: <FormattedMessage defaultMessage="Time" description="Column selector label for the start-time column" />,
  input: <FormattedMessage defaultMessage="Input" description="Column selector label for the input column" />,
  output: <FormattedMessage defaultMessage="Output" description="Column selector label for the output column" />,
  session: <FormattedMessage defaultMessage="Session" description="Column selector label for the session column" />,
  duration: <FormattedMessage defaultMessage="Duration" description="Column selector label for the duration column" />,
  state: <FormattedMessage defaultMessage="State" description="Column selector label for the state column" />,
  trace_id: <FormattedMessage defaultMessage="Trace ID" description="Column selector label for the trace-id column" />,
  tokens: <FormattedMessage defaultMessage="Tokens" description="Column selector label for the tokens column" />,
  cost: <FormattedMessage defaultMessage="Cost" description="Column selector label for the cost column" />,
  tags: <FormattedMessage defaultMessage="Tags" description="Column selector label for the tags column" />,
};

// Static per-column componentIds (the lint rule requires static ids). Canonical render order.
const COLUMN_OPTIONS: ColumnSelectorOption[] = [
  {
    id: 'start_time',
    label: COLUMN_LABELS.start_time,
    componentId: 'mlflow.traces-v4.column-selector.item.start_time',
  },
  { id: 'input', label: COLUMN_LABELS.input, componentId: 'mlflow.traces-v4.column-selector.item.input' },
  { id: 'output', label: COLUMN_LABELS.output, componentId: 'mlflow.traces-v4.column-selector.item.output' },
  { id: 'session', label: COLUMN_LABELS.session, componentId: 'mlflow.traces-v4.column-selector.item.session' },
  { id: 'duration', label: COLUMN_LABELS.duration, componentId: 'mlflow.traces-v4.column-selector.item.duration' },
  { id: 'state', label: COLUMN_LABELS.state, componentId: 'mlflow.traces-v4.column-selector.item.state' },
  { id: 'trace_id', label: COLUMN_LABELS.trace_id, componentId: 'mlflow.traces-v4.column-selector.item.trace_id' },
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
 * Builds the V4-specific controls for the shared `TracesTableToolbar`. Filters and columns stay
 * beside the shared search while a flexible spacer pins Detect Issues and refresh right. None depend
 * on trace data, so they paint immediately.
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
  selectionCount,
  onBulkDelete,
  isDeleteDisabled,
  isRefreshing,
  experimentId,
  actions,
  selectedTraceInfos,
  onDetectIssues,
  savedViewsButton,
}: TracesV4ToolbarParams): TracesV4ToolbarSlots => {
  const intl = useIntl();
  const hasSelection = selectionCount > 0;
  const filterFields = useMlflowTraceFilterFields(assessmentColumns.candidateNames);

  // Siloed copy of v3's UC-delete disabled reason.
  const deleteDisabledReason = intl.formatMessage({
    defaultMessage:
      'Trace deletion is not supported for traces located in Unity Catalog schema. You can delete traces from corresponding Delta table.',
    description:
      'Trace deletion disabled reason. Displayed in a tooltip when a user attempts to delete a trace housed in the UC delta table.',
  });

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
        <TraceColumnSelector
          columns={COLUMN_OPTIONS}
          visibleColumns={visibleColumns}
          onToggleColumn={onToggleColumn}
          onResetToDefaults={onResetColumns}
          groups={
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
              : undefined
          }
        />
        {hasSelection && (
          <TracesV4ActionsButton
            selectionCount={selectionCount}
            onDelete={onBulkDelete}
            disabled={isDeleteDisabled}
            disabledReason={isDeleteDisabled ? deleteDisabledReason : undefined}
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
