import { useMemo, type Ref } from 'react';
import {
  useDesignSystemTheme,
  ArrowsUpDownIcon,
  Button,
  SyncIcon,
  PinCancelIcon,
  PinIcon,
  Tooltip,
  Typography,
} from '@databricks/design-system';
import type { OnChangeFn, RowSelectionState, SortingState } from '@tanstack/react-table';
import { FormattedMessage, FormattedRelativeTime, useIntl } from 'react-intl';
import type { RunEntity } from '../../types';
import { getRunTableMetadata } from './ExperimentEvaluationRunsTable.utils';
import { EVAL_RUNS_TOOLBAR_ICON_SIZE } from './ExperimentEvaluationRunsTable.constants';
import type { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import type { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import { ExperimentEvaluationRunsTableActions } from './ExperimentEvaluationRunsTableActions';
import { EvalRunsFilterButton, type EvalRunsFilterButtonHandle } from './EvalRunsFilterButton';
import { EvalRunsDateFilterButton } from './EvalRunsDateFilterButton';
import { ExperimentEvaluationRunsDisplayButton } from './ExperimentEvaluationRunsDisplayButton';
import { EvalRunsUnifiedSearch } from './EvalRunsUnifiedSearch';
import type { EvalRunsDatePreset, EvalRunsFiltersState } from './hooks/useEvalRunsFilters';
import type { EvalRunsFilterClause } from './evalRunsFilterFields';
import { isMatchingEvalRunsBaseline } from './hooks/useEvalRunsBaseline';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION } from '../../constants';

const SEARCH_MIN_WIDTH = 240;

/**
 * Single-row eval-runs toolbar (V2), replacing the two-row `ExperimentEvaluationRunsTableControls`.
 * Gated behind `shouldEnableNewEvalRunsUi`. Date range + search on the left, then filters and display
 * controls, contextual actions, and a refresh button pinned right. Wraps on narrow panes.
 */
export const ExperimentEvaluationRunsTableControlsV2 = ({
  experimentId,
  rowSelection,
  setRowSelection,
  refetchRuns,
  isFetching,
  lastUpdatedTime,
  runs,
  searchRunsError,
  searchFilter,
  setSearchFilter,
  selectedColumns,
  setSelectedColumns,
  filters,
  setDatePreset,
  setClauses,
  excludeIssueDetectionRuns,
  filterButtonRef,
  groupByConfig,
  setGroupByConfig,
  sorting,
  setSorting,
  onSortingChange,
  viewMode,
  setViewMode,
  onCompare,
  selectedRunUuid,
  compareToRunUuid,
  setIsComparisonMode,
  enableNRunComparison = false,
  onCompareNavigate,
  onSetBaseline,
  onClearBaseline,
  baselineRunUuid,
  isSavingBaseline,
  onFilterChartToSelectedRuns,
  isChartFilteredToSelection,
}: {
  experimentId: string;
  rowSelection: RowSelectionState;
  setRowSelection: (selection: RowSelectionState) => void;
  runs: RunEntity[];
  refetchRuns: () => void;
  isFetching: boolean;
  /** ms timestamp of the last successful runs fetch (react-query `dataUpdatedAt`); drives "Updated X ago". */
  lastUpdatedTime?: number;
  searchRunsError: ErrorWrapper | Error | null;
  searchFilter: string;
  setSearchFilter: (filter: string) => void;
  selectedColumns: { [key: string]: boolean };
  setSelectedColumns: (columns: { [key: string]: boolean }) => void;
  filters: EvalRunsFiltersState;
  setDatePreset: (preset: EvalRunsDatePreset) => void;
  setClauses: (clauses: EvalRunsFilterClause[]) => void;
  excludeIssueDetectionRuns: boolean;
  filterButtonRef?: Ref<EvalRunsFilterButtonHandle>;
  groupByConfig: RunsGroupByConfig | null;
  setGroupByConfig: (groupBy: RunsGroupByConfig | null) => void;
  sorting?: SortingState;
  setSorting?: OnChangeFn<SortingState>;
  /** Alias for callers that use TanStack's controlled-table naming. */
  onSortingChange?: OnChangeFn<SortingState>;
  viewMode?: ExperimentEvaluationRunsPageMode;
  setViewMode?: (mode: ExperimentEvaluationRunsPageMode) => void;
  onCompare: (runUuid1: string, runUuid2: string) => void;
  selectedRunUuid?: string;
  compareToRunUuid?: string;
  setIsComparisonMode: (isComparisonMode: boolean) => void;
  /** When true, render a standalone "Compare (N)" button that navigates to the compare page. */
  enableNRunComparison?: boolean;
  onCompareNavigate?: (runUuids: string[]) => void;
  onSetBaseline?: (runUuid: string) => void;
  onClearBaseline?: () => void;
  baselineRunUuid?: string;
  isSavingBaseline?: boolean;
  onFilterChartToSelectedRuns?: () => void;
  isChartFilteredToSelection?: boolean;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const selectedRunUuids = Object.entries(rowSelection)
    .filter(([_, value]) => value)
    .map(([key]) => key);
  const hasSelection = selectedRunUuids.length > 0;
  const selectedBaselineCandidateUuid = selectedRunUuids.length === 1 ? selectedRunUuids[0] : undefined;
  const selectedRunIsBaseline = selectedBaselineCandidateUuid === baselineRunUuid;
  const isSelectedRunIssueDetection = runs
    .find((run) => run.info.runUuid === selectedBaselineCandidateUuid)
    // The OSS server omits empty repeated fields, so a run without tags has no `tags` array.
    ?.data?.tags?.some(
      ({ key, value }) => key === MLFLOW_RUN_TYPE_TAG && value === MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
    );
  const baselineActionLabel = selectedRunIsBaseline
    ? intl.formatMessage({
        defaultMessage: 'Clear baseline',
        description: 'Label for clearing the selected baseline run',
      })
    : baselineRunUuid
      ? intl.formatMessage({
          defaultMessage: 'Replace baseline',
          description: 'Label for replacing the baseline run with the selected run',
        })
      : intl.formatMessage({
          defaultMessage: 'Set baseline',
          description: 'Label for setting the selected run as the baseline',
        });
  const isSelectedRunAvailable = useMemo(() => {
    if (!selectedBaselineCandidateUuid) {
      return true;
    }
    return isMatchingEvalRunsBaseline(
      runs.find((run) => run.info.runUuid === selectedBaselineCandidateUuid),
      selectedBaselineCandidateUuid,
      experimentId,
    );
  }, [experimentId, runs, selectedBaselineCandidateUuid]);

  // Used by the unified search and its filter editor.
  const runsData = useMemo(() => getRunTableMetadata(runs), [runs]);

  const renderEvalRunsSearchBox = (openFilter: (filterIndex?: number) => void) => {
    return (
      <EvalRunsUnifiedSearch
        runs={runs}
        runsData={runsData}
        searchFilter={searchFilter}
        onSearchFilterChange={setSearchFilter}
        clauses={filters.clauses}
        setClauses={setClauses}
        openFilter={openFilter}
        requestError={searchRunsError}
      />
    );
  };

  // flex-wrap: search compresses to SEARCH_MIN_WIDTH first; fixed controls then reflow onto additional
  // lines. Refresh may land alone and right-aligned on a wrapped line — accepted rough edge.
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        flexWrap: 'wrap',
      }}
    >
      <EvalRunsDateFilterButton datePreset={filters.datePreset} setDatePreset={setDatePreset} />

      <div css={{ display: 'flex', flex: 1, minWidth: SEARCH_MIN_WIDTH, maxWidth: 640 }}>
        <EvalRunsFilterButton
          clauses={filters.clauses}
          setClauses={setClauses}
          runsData={runsData}
          excludeIssueDetectionRuns={excludeIssueDetectionRuns}
          filterButtonRef={filterButtonRef}
          renderTrigger={renderEvalRunsSearchBox}
        />
      </div>

      {/* flexShrink:0 keeps this control cluster intact when the row wraps. */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          flexShrink: 0,
        }}
      >
        <ExperimentEvaluationRunsDisplayButton
          runs={runs}
          selectedColumns={selectedColumns}
          setSelectedColumns={setSelectedColumns}
          groupByConfig={groupByConfig}
          setGroupByConfig={setGroupByConfig}
          sorting={sorting}
          setSorting={onSortingChange ?? setSorting}
          viewMode={viewMode}
          setViewMode={setViewMode}
        />

        {selectedBaselineCandidateUuid &&
          ((selectedRunIsBaseline && onClearBaseline) || (!selectedRunIsBaseline && onSetBaseline)) && (
            <Tooltip
              componentId="mlflow.eval-runs.baseline-action.tooltip"
              content={
                isSelectedRunIssueDetection && !selectedRunIsBaseline ? (
                  <FormattedMessage
                    defaultMessage="Issue detection runs cannot be used as a baseline"
                    description="Tooltip explaining why an issue detection run cannot be set as the baseline"
                  />
                ) : (
                  baselineActionLabel
                )
              }
            >
              <Button
                componentId="mlflow.eval-runs.baseline-action"
                aria-label={baselineActionLabel}
                css={{
                  flexShrink: 0,
                  '&&&': {
                    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px !important`,
                  },
                }}
                disabled={
                  isSavingBaseline ||
                  (!selectedRunIsBaseline && (!isSelectedRunAvailable || isSelectedRunIssueDetection))
                }
                onClick={() => {
                  if (selectedRunIsBaseline) {
                    onClearBaseline?.();
                  } else {
                    onSetBaseline?.(selectedBaselineCandidateUuid);
                  }
                  setRowSelection({});
                }}
              >
                {selectedRunIsBaseline ? (
                  <PinCancelIcon
                    css={{
                      fontSize: EVAL_RUNS_TOOLBAR_ICON_SIZE,
                      color: theme.colors.textPlaceholder,
                    }}
                  />
                ) : (
                  <PinIcon
                    css={{
                      fontSize: EVAL_RUNS_TOOLBAR_ICON_SIZE,
                      color: theme.colors.textPlaceholder,
                    }}
                  />
                )}
              </Button>
            </Tooltip>
          )}

        {hasSelection && !enableNRunComparison && (
          <ExperimentEvaluationRunsTableActions
            rowSelection={rowSelection}
            setRowSelection={setRowSelection}
            refetchRuns={refetchRuns}
            onCompare={onCompare}
            selectedRunUuid={selectedRunUuid}
            compareToRunUuid={compareToRunUuid}
            foldCompareIntoMenu
            setIsComparisonMode={setIsComparisonMode}
            onFilterChartToSelectedRuns={onFilterChartToSelectedRuns}
            isChartFilteredToSelection={isChartFilteredToSelection}
          />
        )}
      </div>

      {/* Refresh pinned right (marginLeft:auto): spinner while fetching, "X ago" label when settled. */}
      <Tooltip
        componentId="mlflow.eval-runs.table-refresh-button.tooltip"
        content={
          lastUpdatedTime ? (
            <FormattedMessage
              defaultMessage="Updated {time}"
              description="Tooltip for the eval-runs refresh button showing how long ago the runs were last updated"
              values={{
                time: (
                  <FormattedRelativeTime
                    value={(lastUpdatedTime - Date.now()) / 1000}
                    numeric="always"
                    updateIntervalInSeconds={10}
                  />
                ),
              }}
            />
          ) : (
            intl.formatMessage({
              defaultMessage: 'Refresh evaluation runs',
              description: 'Tooltip for the refresh evaluation runs button in the eval-runs toolbar',
            })
          )
        }
      >
        <Button
          componentId="mlflow.eval-runs.table-refresh-button"
          type="tertiary"
          onClick={refetchRuns}
          disabled={isFetching}
          css={{ flexShrink: 0, marginLeft: 'auto' }}
          aria-label={intl.formatMessage({
            defaultMessage: 'Refresh evaluation runs',
            description: 'Aria label for the refresh evaluation runs button in the eval-runs toolbar',
          })}
        >
          {/* size="sm" via prop — Text's doubled-selector specificity beats a css fontSize override. */}
          <Typography.Text
            size="sm"
            color="secondary"
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
            }}
          >
            {isFetching ? (
              <SyncIcon
                spin
                css={{
                  fontSize: EVAL_RUNS_TOOLBAR_ICON_SIZE,
                  color: theme.colors.textPlaceholder,
                }}
              />
            ) : (
              <>
                <SyncIcon
                  css={{
                    fontSize: EVAL_RUNS_TOOLBAR_ICON_SIZE,
                    color: theme.colors.textPlaceholder,
                  }}
                />
                {lastUpdatedTime ? (
                  <FormattedRelativeTime
                    value={(lastUpdatedTime - Date.now()) / 1000}
                    numeric="auto"
                    updateIntervalInSeconds={10}
                  />
                ) : null}
              </>
            )}
          </Typography.Text>
        </Button>
      </Tooltip>

      {enableNRunComparison && hasSelection && (
        <ExperimentEvaluationRunsTableActions
          rowSelection={rowSelection}
          setRowSelection={setRowSelection}
          refetchRuns={refetchRuns}
          onCompare={onCompare}
          selectedRunUuid={selectedRunUuid}
          compareToRunUuid={compareToRunUuid}
          foldCompareIntoMenu
          setIsComparisonMode={setIsComparisonMode}
          enableNewCompare
          onFilterChartToSelectedRuns={onFilterChartToSelectedRuns}
          isChartFilteredToSelection={isChartFilteredToSelection}
        />
      )}

      {enableNRunComparison && hasSelection && (
        <Tooltip
          componentId="mlflow.eval-runs.compare-navigate-button.tooltip"
          content={
            selectedRunUuids.length >= 2 ? (
              <FormattedMessage
                defaultMessage="Compare selected runs"
                description="Tooltip for the eval-runs compare button when at least 2 runs are selected"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Select at least 2 runs to compare"
                description="Tooltip for the eval-runs compare button when fewer than 2 runs are selected"
              />
            )
          }
        >
          <Button
            componentId="mlflow.eval-runs.compare-navigate-button"
            type="primary"
            icon={<ArrowsUpDownIcon rotate={90} />}
            disabled={selectedRunUuids.length < 2}
            onClick={() => onCompareNavigate?.(selectedRunUuids)}
          >
            <FormattedMessage
              defaultMessage="Compare ({numRuns})"
              description="Standalone eval-runs compare button label with the number of selected runs"
              values={{ numRuns: selectedRunUuids.length }}
            />
          </Button>
        </Tooltip>
      )}
    </div>
  );
};
