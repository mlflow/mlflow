import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  ForkHorizontalIcon,
  SchemaIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  SpeechBubbleIcon,
  ToggleButton,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { DetectIssuesButton, shouldEnableSessionGrouping } from '@databricks/web-shared/genai-traces-table';
import {
  TraceFilterButton,
  type TraceFilterButtonProps,
  type ColumnSelectorOption,
  type FilterFieldDef,
  type SortDirection,
  type ReorderableTraceColumnOption,
  type TraceColumnId,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { shouldEnableIssueDetection } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { CancelIssueDetectionButton } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-issues/CancelIssueDetectionButton';
import { type TracesV4AssessmentColumns } from '../hooks/useTracesV4AssessmentColumns';
import { type TracesV4CustomColumns } from '../hooks/useTracesV4CustomColumns';
import { type TracesV4Density } from '../hooks/useTracesV4Density';
import { isAssessmentColumnId } from '../utils/assessmentColumns';
import { TracesV4DateSelector, TracesV4RefreshButton } from './TracesV4DateSelector';
import { TracesV4DisplayButton, type TracesV4ColumnGroup } from './TracesV4DisplayButton';
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
  /** Full display order across standard + assessment columns; drives the reorderable list order. */
  columnOrder: string[];
  /** Reorders a column (drag or Ctrl+Arrow) in the column selector; moves `active` to `target`. */
  onReorderColumn: (activeColumn: string, targetColumn: string) => void;
  /** Assessment column selection (dynamic, per-page) rendered as a group in the column selector. */
  assessmentColumns: TracesV4AssessmentColumns;
  /** Custom (tag + metadata) column selection (dynamic, per-page). Omit to offer no custom columns. */
  customColumns?: TracesV4CustomColumns;
  /** Active sort column + direction, and its setter — surfaced in the Display popover's Sort submenu. */
  sort: TraceColumnId;
  dir: SortDirection;
  onSort: (column: TraceColumnId, direction: SortDirection) => void;
  /** Row-height density + setter — surfaced in the Display popover's Row height submenu. */
  density: TracesV4Density;
  onDensityChange: (density: TracesV4Density) => void;
  /** Whether traces are grouped into collapsible session rows. */
  isGroupedBySession: boolean;
  /** Toggle session grouping on/off. */
  onToggleSessionGrouping: (next: boolean) => void;
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
  /**
   * A detection already in flight, which the button offers to cancel rather than starting a
   * second one. Omit when none is running. `isStarting` covers the gap before the run exists:
   * the button spins, since there is nothing to cancel yet.
   */
  inProgressIssueDetection?: { isStarting: boolean; onCancel?: () => Promise<void> };
  /** The "Views" saved-views dropdown, rendered far left (before the date selector). Omit to hide it. */
  savedViewsButton?: React.ReactNode;
  /**
   * Leave the Filters button and the spacer after it out of the right controls, for a toolbar whose
   * search renders the filter editor itself (through `renderFilterEditor`).
   */
  hideFilterButton?: boolean;
  /** Traces on the current page; with the new evaluation runs UI the filter editor suggests their values. */
  traces?: ModelTraceInfoV3[];
  /** Replaces the Filters button's fields (default: the trace filter fields for these traces). */
  filterFields?: FilterFieldDef[];
  /** Explains the Filters button's matching semantics (default: none). */
  filterDescription?: TraceFilterButtonProps['description'];
  /** Rejects an Apply in the Filters button while keeping the draft editable (default: none). */
  validateFilters?: TraceFilterButtonProps['validateFilters'];
  /** Offer the session grouping toggle (default true, when session grouping is enabled). */
  showSessionGrouping?: boolean;
  /** Columns the table force-shows while grouped by session, locked in the Display column list (default none). */
  groupedLockedColumns?: readonly TraceColumnId[];
  /** Show a Traces/Sessions switch instead of the group-by-session toggle (default false). */
  showSessionViewSwitch?: boolean;
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
  metadata: (
    <FormattedMessage defaultMessage="Metadata" description="Column selector label for the aggregate metadata column" />
  ),
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
  { id: 'metadata', label: COLUMN_LABELS.metadata, componentId: 'mlflow.traces-v4.column-selector.item.metadata' },
];

export interface TracesV4ToolbarSlots {
  /** The filter editor anchored to a consumer-rendered trigger, such as the unified search box. */
  renderFilterEditor: (
    renderTrigger: (openEditor: (filterIndex?: number) => void) => React.ReactNode,
  ) => React.ReactNode;
  /** Rendered before the shared search box: the Views dropdown, the Traces/Sessions switch, then the date range. */
  leftControls: React.ReactNode;
  /** Rendered after the shared search box: filters, columns, a spacer, Detect Issues, and refresh. */
  rightControls: React.ReactNode;
  /** The Traces/Sessions switch alone, for tables whose left slot has no Views dropdown or date range. */
  sessionViewSwitch: React.ReactNode;
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
  customColumns,
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
  inProgressIssueDetection,
  savedViewsButton,
  hideFilterButton = false,
  traces,
  filterFields: filterFieldsOverride,
  filterDescription,
  validateFilters,
  showSessionGrouping = true,
  groupedLockedColumns,
  showSessionViewSwitch = false,
  isGroupedBySession,
  onToggleSessionGrouping,
  columnOrder,
  onReorderColumn,
}: TracesV4ToolbarParams): TracesV4ToolbarSlots => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const hasSelection = selectionCount > 0;
  const tracesViewLabel = intl.formatMessage({
    defaultMessage: 'Traces',
    description: 'Trace table view switcher option',
  });
  const sessionsViewLabel = intl.formatMessage({
    defaultMessage: 'Sessions',
    description: 'Session table view switcher option',
  });
  const defaultFilterFields = useMlflowTraceFilterFields(assessmentColumns.candidateNames, traces);
  const filterFields = filterFieldsOverride ?? defaultFilterFields;

  // While grouped by session the table force-shows the Session column, so lock its column-selector
  // toggle (checked + disabled) rather than letting an unchecking click silently no-op.
  const columnOptions = isGroupedBySession
    ? COLUMN_OPTIONS.map((option) => (option.id === 'session' ? { ...option, disabled: true } : option))
    : COLUMN_OPTIONS;

  // Plain-text label per standard column for the drag handle's aria-label (a `reorderLabel` must be a
  // string). Assessment columns use their name directly. Kept beside `COLUMN_LABELS` so both stay in
  // sync when a column is added.
  const columnDisplayText = useMemo<Record<TraceColumnId, string>>(
    () => ({
      start_time: intl.formatMessage({ defaultMessage: 'Time', description: 'Traces table start-time column name' }),
      input: intl.formatMessage({ defaultMessage: 'Input', description: 'Traces table input column name' }),
      output: intl.formatMessage({ defaultMessage: 'Output', description: 'Traces table output column name' }),
      session: intl.formatMessage({ defaultMessage: 'Session', description: 'Traces table session column name' }),
      duration: intl.formatMessage({ defaultMessage: 'Duration', description: 'Traces table duration column name' }),
      state: intl.formatMessage({ defaultMessage: 'State', description: 'Traces table state column name' }),
      trace_id: intl.formatMessage({ defaultMessage: 'Trace ID', description: 'Traces table trace-id column name' }),
      trace_name: intl.formatMessage({
        defaultMessage: 'Trace name',
        description: 'Traces table trace-name column name',
      }),
      user: intl.formatMessage({ defaultMessage: 'User', description: 'Traces table user column name' }),
      source: intl.formatMessage({ defaultMessage: 'Source', description: 'Traces table source column name' }),
      run_name: intl.formatMessage({ defaultMessage: 'Run name', description: 'Traces table run-name column name' }),
      tokens: intl.formatMessage({ defaultMessage: 'Tokens', description: 'Traces table tokens column name' }),
      cost: intl.formatMessage({ defaultMessage: 'Cost', description: 'Traces table cost column name' }),
      tags: intl.formatMessage({ defaultMessage: 'Tags', description: 'Traces table tags column name' }),
      metadata: intl.formatMessage({ defaultMessage: 'Metadata', description: 'Traces table metadata column name' }),
    }),
    [intl],
  );

  // The unified reorderable list: standard + assessment columns in the persisted `columnOrder`, each
  // carrying a localized reorder aria-label. Standard columns keep their rich `FormattedMessage`
  // label; assessment columns use their name. Ids unknown on this page (a stored assessment not in the
  // current data) are dropped by the flatMap.
  const reorderableColumns = useMemo<ReorderableTraceColumnOption[]>(() => {
    // Force-shown columns stay checked and disabled, so unchecking them can't silently no-op.
    const lockedColumns = isGroupedBySession ? groupedLockedColumns : undefined;
    const standardById = new Map(
      COLUMN_OPTIONS.map((option) => [
        option.id,
        lockedColumns?.includes(option.id) ? { ...option, disabled: true } : option,
      ]),
    );
    const assessmentById = new Map(assessmentColumns.selectorOptions.map((option) => [option.id, option]));
    const reorderLabelFor = (name: string) =>
      intl.formatMessage(
        {
          defaultMessage: 'Reorder {column} column',
          description: 'Aria label for the drag handle that reorders a traces table column',
        },
        { column: name },
      );
    return columnOrder.flatMap((id) => {
      const standard = standardById.get(id as TraceColumnId);
      if (standard) {
        return [{ ...standard, reorderLabel: reorderLabelFor(columnDisplayText[id as TraceColumnId]) }];
      }
      const assessment = assessmentById.get(id);
      if (assessment) {
        return [{ ...assessment, reorderLabel: reorderLabelFor(String(assessment.label)) }];
      }
      return [];
    });
  }, [
    columnOrder,
    assessmentColumns.selectorOptions,
    columnDisplayText,
    intl,
    isGroupedBySession,
    groupedLockedColumns,
  ]);

  // A single toggle that routes to the right handler by id: assessment ids go to the assessment
  // toggle, standard ids to the column toggle. The reorderable list is a flat list of both.
  const toggleAnyColumn = (id: string) =>
    isAssessmentColumnId(id) ? assessmentColumns.toggle(id) : onToggleColumn(id as TraceColumnId);
  const visibleAnyColumns = [...visibleColumns, ...assessmentColumns.visibleIds];

  // Siloed copy of v3's UC-delete disabled reason.
  const deleteDisabledReason = intl.formatMessage({
    defaultMessage:
      'Trace deletion is not supported for traces located in Unity Catalog schema. You can delete traces from corresponding Delta table.',
    description:
      'Trace deletion disabled reason. Displayed in a tooltip when a user attempts to delete a trace housed in the UC delta table.',
  });

  // Dynamic (per-page) columns render as labeled checkbox groups beneath the reorderable list in the
  // Display → Columns submenu. Tags/metadata are toggle-only. Assessments ALSO appear in the
  // reorderable list above (so they can be dragged + toggled individually); the group here adds only
  // the bulk "show/hide all assessments" affordance via its toggle-all header. Each group is included
  // only when the page has relevant candidates.
  const columnGroups: TracesV4ColumnGroup[] = [];
  if (customColumns && customColumns.tags.selectorOptions.length > 0) {
    columnGroups.push({
      label: (
        <FormattedMessage
          defaultMessage="Tags"
          description="Section label for tag columns in the traces table column selector"
        />
      ),
      options: customColumns.tags.selectorOptions,
      visibleIds: customColumns.tags.visibleIds,
      onToggle: customColumns.tags.toggle,
      onToggleAll: customColumns.tags.setAllVisible,
    });
  }
  if (customColumns && customColumns.metadata.selectorOptions.length > 0) {
    columnGroups.push({
      label: (
        <FormattedMessage
          defaultMessage="Metadata"
          description="Section label for metadata columns in the traces table column selector"
        />
      ),
      options: customColumns.metadata.selectorOptions,
      visibleIds: customColumns.metadata.visibleIds,
      onToggle: customColumns.metadata.toggle,
      onToggleAll: customColumns.metadata.setAllVisible,
    });
  }
  if (assessmentColumns.selectorOptions.length > 0) {
    columnGroups.push({
      label: (
        <FormattedMessage
          defaultMessage="Assessments"
          description="Section label for assessment columns in the traces table column selector"
        />
      ),
      options: assessmentColumns.selectorOptions,
      visibleIds: assessmentColumns.visibleIds,
      onToggle: assessmentColumns.toggle,
      onToggleAll: assessmentColumns.setAllVisible,
      // Assessments already appear (draggable + individually toggleable) in the reorderable list
      // above; this group adds only the bulk "show/hide all" header, so skip its per-item checkboxes.
      headerOnly: true,
    });
  }

  // Shown before the search box, after the Views dropdown where there is one.
  const sessionViewSwitch =
    showSessionGrouping && shouldEnableSessionGrouping() && showSessionViewSwitch ? (
      <SegmentedControlGroup
        name="mlflow.traces-v4.table-view"
        componentId="mlflow.traces-v4.table-view"
        value={isGroupedBySession ? 'sessions' : 'traces'}
        newStyleFlagOverride
      >
        <SegmentedControlButton
          value="traces"
          aria-label={tracesViewLabel}
          onClick={() => onToggleSessionGrouping(false)}
          icon={
            <Tooltip
              componentId="mlflow.traces-v4.table-view-traces-tooltip"
              content={tracesViewLabel}
              delayDuration={0}
            >
              <ForkHorizontalIcon />
            </Tooltip>
          }
        />
        <SegmentedControlButton
          value="sessions"
          aria-label={sessionsViewLabel}
          onClick={() => onToggleSessionGrouping(true)}
          icon={
            <Tooltip
              componentId="mlflow.traces-v4.table-view-sessions-tooltip"
              content={sessionsViewLabel}
              delayDuration={0}
            >
              <SpeechBubbleIcon />
            </Tooltip>
          }
        />
      </SegmentedControlGroup>
    ) : null;

  // Note: the Databricks build disables Delete for UC-backed traces (with a "delete from the Delta
  // table instead" tooltip). OSS traces are always deletable via the standard path, so that gate is
  // dropped here and Delete is always enabled.
  return {
    leftControls: (
      <>
        {savedViewsButton}
        {sessionViewSwitch}
        <TracesV4DateSelector experimentId={experimentId} />
      </>
    ),
    sessionViewSwitch,
    renderFilterEditor: (renderTrigger) => (
      <TraceFilterButton
        fields={filterFields}
        filterModel={filterModel}
        onChange={onFilterChange}
        onClearAll={onClearFilters}
        activeCount={activeFilterCount}
        renderTrigger={renderTrigger}
        description={filterDescription}
        validateFilters={validateFilters}
      />
    ),
    rightControls: (
      <>
        {!hideFilterButton && (
          <TraceFilterButton
            fields={filterFields}
            filterModel={filterModel}
            onChange={onFilterChange}
            onClearAll={onClearFilters}
            activeCount={activeFilterCount}
            description={filterDescription}
            validateFilters={validateFilters}
          />
        )}
        {showSessionGrouping && shouldEnableSessionGrouping() && !showSessionViewSwitch && (
          <Tooltip
            componentId="mlflow.traces-v4.group-by-session.tooltip"
            content={intl.formatMessage({
              defaultMessage: 'Group traces by session',
              description: 'Tooltip for the group by session button in the V4 traces table toolbar',
            })}
          >
            <ToggleButton
              componentId="mlflow.traces-v4.group-by-session"
              pressed={isGroupedBySession}
              onPressedChange={onToggleSessionGrouping}
              icon={<SchemaIcon />}
              // Icon-only ToggleButtons render borderless here; managed's `forceWithBorder` prop
              // doesn't exist in this design-system version, so restore the border manually.
              css={{
                border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
                boxShadow: theme.shadows.xs,
              }}
              aria-label={intl.formatMessage({
                defaultMessage: 'Group traces by session',
                description: 'Accessible label for the group by session button in the V4 traces table toolbar',
              })}
            />
          </Tooltip>
        )}
        <TracesV4DisplayButton
          onResetColumns={onResetColumns}
          sortColumnLabels={COLUMN_LABELS}
          sort={sort}
          dir={dir}
          onSort={onSort}
          density={density}
          onDensityChange={onDensityChange}
          reorder={{
            columns: reorderableColumns,
            visibleColumns: visibleAnyColumns,
            onToggleColumn: toggleAnyColumn,
            onReorderColumn,
          }}
          columnGroups={columnGroups}
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
        {!hideFilterButton && <div css={{ flex: 1 }} />}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <TracesV4RefreshButton isFetching={isRefreshing} />
          {shouldEnableIssueDetection() &&
            onDetectIssues &&
            (inProgressIssueDetection && !inProgressIssueDetection.isStarting ? (
              <CancelIssueDetectionButton
                componentId="mlflow.traces-v4.cancel-issue-detection-button"
                onCancel={inProgressIssueDetection.onCancel}
              />
            ) : (
              <DetectIssuesButton
                componentId="mlflow.traces-v4.detect-issues-button"
                onClick={onDetectIssues}
                loading={Boolean(inProgressIssueDetection?.isStarting)}
              />
            ))}
        </div>
      </>
    ),
  };
};
