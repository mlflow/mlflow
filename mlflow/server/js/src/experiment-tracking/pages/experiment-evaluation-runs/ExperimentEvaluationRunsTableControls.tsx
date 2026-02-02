import {
  useDesignSystemTheme,
  Button,
  DialogCombobox,
  DialogComboboxTrigger,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  RefreshIcon,
  DialogComboboxSectionHeader,
  Spacer,
  SegmentedControlGroup,
  SegmentedControlButton,
  TableIcon,
  ChartLineIcon,
  ListBorderIcon,
  Tooltip,
} from '@databricks/design-system';
import type { RowSelectionState } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';
import { RunsSearchAutoComplete } from '../../components/experiment-page/components/runs/RunsSearchAutoComplete';
import type { RunEntity } from '../../types';
import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import type { KeyValueEntity } from '../../../common/types';
import type { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';
import { useCallback, useMemo } from 'react';
import type { EvalRunsTableColumnId } from './ExperimentEvaluationRunsTable.constants';
import {
  EVAL_RUNS_COLUMN_LABELS,
  EVAL_RUNS_COLUMN_TYPE_LABELS,
  EVAL_RUNS_UNSELECTABLE_COLUMNS,
  EvalRunsTableKeyedColumnPrefix,
} from './ExperimentEvaluationRunsTable.constants';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { groupBy } from 'lodash';
import { ExperimentEvaluationRunsTableGroupBySelector } from './ExperimentEvaluationRunsTableGroupBySelector';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import { ExperimentEvaluationRunsTableActions } from './ExperimentEvaluationRunsTableActions';
import { shouldEnableImprovedEvalRunsComparison } from '../../../common/utils/FeatureUtils';

// function to mimic the data structure of the legacy runs response
// so we can reuse the RunsSearchAutoComplete component
const getRunTableMetadata = (runsData: RunEntity[]): ExperimentRunsSelectorResult => {
  const metricKeys = new Set<string>();
  const paramKeys = new Set<string>();
  const tags: Record<string, KeyValueEntity>[] = [];

  runsData.forEach((run) => {
    run.data.metrics?.forEach((metric) => {
      metricKeys.add(metric.key);
    });
    run.data.params?.forEach((param) => {
      paramKeys.add(param.key);
    });

    const runTags: Record<string, KeyValueEntity> = {};
    run.data.tags?.forEach((tag) => {
      runTags[tag.key] = { key: tag.key, value: tag.value };
    });

    tags.push(runTags);
  });

  return {
    metricKeyList: Array.from(metricKeys),
    paramKeyList: Array.from(paramKeys),
    tagsList: tags,
  } as ExperimentRunsSelectorResult;
};

export const ExperimentEvaluationRunsTableControls = ({
  rowSelection,
  setRowSelection,
  refetchRuns,
  isFetching,
  runs,
  searchRunsError,
  searchFilter,
  setSearchFilter,
  selectedColumns,
  setSelectedColumns,
  groupByConfig,
  setGroupByConfig,
  viewMode,
  setViewMode,
  onCompare,
  selectedRunUuid,
  compareToRunUuid,
  isComparisonMode,
  setIsComparisonMode,
}: {
  rowSelection: RowSelectionState;
  setRowSelection: (selection: RowSelectionState) => void;
  runs: RunEntity[];
  refetchRuns: () => void;
  isFetching: boolean;
  searchRunsError: ErrorWrapper | Error | null;
  searchFilter: string;
  setSearchFilter: (filter: string) => void;
  selectedColumns: { [key: string]: boolean };
  setSelectedColumns: (columns: { [key: string]: boolean }) => void;
  groupByConfig: RunsGroupByConfig | null;
  setGroupByConfig: (groupBy: RunsGroupByConfig | null) => void;
  viewMode?: ExperimentEvaluationRunsPageMode;
  setViewMode?: (mode: ExperimentEvaluationRunsPageMode) => void;
  onCompare: (runUuid1: string, runUuid2: string) => void;
  selectedRunUuid?: string;
  compareToRunUuid?: string;
  isComparisonMode: boolean;
  setIsComparisonMode: (isComparisonMode: boolean) => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const selectedRunUuids = Object.entries(rowSelection)
    .filter(([_, value]) => value)
    .map(([key]) => key);

  const columnPartitions = useMemo(
    () =>
      groupBy(
        Object.entries(selectedColumns),
        ([columnId]) =>
          parseEvalRunsTableKeyedColumnKey(columnId)?.columnType ?? EvalRunsTableKeyedColumnPrefix.ATTRIBUTE,
      ),
    [selectedColumns],
  );

  const isCompareEnabled = selectedRunUuids.length >= 1;

  const handleCompareClick = useCallback(() => {
    if (selectedRunUuids.length >= 1) {
      if (selectedRunUuids.length >= 2) {
        onCompare(selectedRunUuids[0], selectedRunUuids[1]);
      }
      setIsComparisonMode(true);
    }
  }, [selectedRunUuids, onCompare, setIsComparisonMode]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <SegmentedControlGroup
          name="mlflow.eval-runs.page-mode-selector"
          componentId="mlflow.eval-runs.page-mode-selector"
          value={viewMode}
          css={{ flexShrink: 0 }}
        >
          <SegmentedControlButton
            value={ExperimentEvaluationRunsPageMode.TRACES}
            icon={
              <Tooltip
                componentId="mlflow.eval-runs.traces-mode-toggle-tooltip"
                content={
                  <FormattedMessage
                    defaultMessage="Trace view"
                    description="Tooltip for traces preview mode toggle in evaluation runs table controls"
                  />
                }
                delayDuration={0}
              >
                <ListBorderIcon />
              </Tooltip>
            }
            onClick={() => setViewMode?.(ExperimentEvaluationRunsPageMode.TRACES)}
          />
          <SegmentedControlButton
            value={ExperimentEvaluationRunsPageMode.CHARTS}
            icon={
              <Tooltip
                componentId="mlflow.eval-runs.charts-mode-toggle-tooltip"
                content={
                  <FormattedMessage
                    defaultMessage="Charts"
                    description="Tooltip for charts page mode toggle in evaluation runs table controls"
                  />
                }
                delayDuration={0}
              >
                <ChartLineIcon />
              </Tooltip>
            }
            onClick={() => setViewMode?.(ExperimentEvaluationRunsPageMode.CHARTS)}
          />
        </SegmentedControlGroup>
        <RunsSearchAutoComplete
          css={{ minWidth: 0 }}
          runsData={getRunTableMetadata(runs)}
          searchFilter={searchFilter}
          onSearchFilterChange={setSearchFilter}
          onClear={() => setSearchFilter('')}
          requestError={searchRunsError}
        />
        <Tooltip
          componentId="mlflow.eval-runs.table-refresh-button.tooltip"
          content={intl.formatMessage({
            defaultMessage: 'Refresh evaluation runs',
            description: 'Tooltip for the refresh evaluation runs button in the evaluation runs table controls',
          })}
        >
          <Button
            componentId="mlflow.eval-runs.table-refresh-button"
            icon={<RefreshIcon />}
            onClick={refetchRuns}
            loading={isFetching}
            css={{ flexShrink: 0 }}
            disabled={isFetching}
            aria-label={intl.formatMessage({
              defaultMessage: 'Refresh evaluation runs',
              description: 'Aria label for the refresh evaluation runs button in the evaluation runs table controls',
            })}
          />
        </Tooltip>
      </div>
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <DialogCombobox componentId="mlflow.eval-runs.table-column-selector" label="Columns" multiSelect>
          <DialogComboboxTrigger />
          <DialogComboboxContent>
            <DialogComboboxOptionList>
              {Object.entries(columnPartitions).map(([columnType, columns]) => {
                if (!columns.length) {
                  return null;
                }
                const headerLabelDescriptor =
                  EVAL_RUNS_COLUMN_TYPE_LABELS[columnType as EvalRunsTableKeyedColumnPrefix];
                return (
                  // eslint-disable-next-line react/jsx-key
                  <>
                    <Spacer size="xs" />
                    <DialogComboboxSectionHeader>
                      {headerLabelDescriptor ? intl.formatMessage(headerLabelDescriptor) : columnType}
                    </DialogComboboxSectionHeader>
                    {columns.map(([column, selected]) => {
                      const labelDescriptorForKnownColumn = EVAL_RUNS_COLUMN_LABELS[column as EvalRunsTableColumnId];
                      const label = labelDescriptorForKnownColumn
                        ? intl.formatMessage(labelDescriptorForKnownColumn)
                        : (parseEvalRunsTableKeyedColumnKey(column)?.key ?? column);

                      if (EVAL_RUNS_UNSELECTABLE_COLUMNS.has(column)) {
                        return null;
                      }

                      return (
                        <DialogComboboxOptionListCheckboxItem
                          key={column}
                          value={column}
                          onChange={() => {
                            const newSelectedColumns = { ...selectedColumns };
                            newSelectedColumns[column] = !selected;
                            setSelectedColumns(newSelectedColumns);
                          }}
                          checked={selected}
                        >
                          {label}
                        </DialogComboboxOptionListCheckboxItem>
                      );
                    })}
                  </>
                );
              })}
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
        <ExperimentEvaluationRunsTableGroupBySelector
          groupByConfig={groupByConfig}
          setGroupByConfig={setGroupByConfig}
          runs={runs}
        />

        {shouldEnableImprovedEvalRunsComparison() && (
          <Tooltip
            componentId="mlflow.eval-runs.compare-button.tooltip"
            content={
              isCompareEnabled ? (
                <FormattedMessage
                  defaultMessage="Compare selected runs"
                  description="Tooltip for the compare button when enabled"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Select up to 2 runs to compare"
                  description="Tooltip for the compare button when disabled"
                />
              )
            }
          >
            <Button
              componentId="mlflow.eval-runs.compare-button"
              onClick={handleCompareClick}
              disabled={!isCompareEnabled}
            >
              <FormattedMessage defaultMessage="Compare" description="Compare runs button label" />
            </Button>
          </Tooltip>
        )}

        <ExperimentEvaluationRunsTableActions
          rowSelection={rowSelection}
          setRowSelection={setRowSelection}
          refetchRuns={refetchRuns}
        />
      </div>
    </div>
  );
};
