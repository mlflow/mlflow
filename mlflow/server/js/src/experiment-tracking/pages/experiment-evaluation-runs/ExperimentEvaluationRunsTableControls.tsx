import {
  useDesignSystemTheme,
  Button,
  Modal,
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
import { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';
import { useCallback, useMemo, useState } from 'react';
import { useDeleteRuns } from '../../components/experiment-page/hooks/useDeleteRuns';
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
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);

  const selectedRunUuids = Object.entries(rowSelection)
    .filter(([_, value]) => value)
    .map(([key]) => key);

  const { mutate, isLoading } = useDeleteRuns({
    onSuccess: () => {
      refetchRuns();
      setRowSelection({});
      setDeleteModalVisible(false);
    },
  });

  const handleDelete = useCallback(() => {
    mutate({ runUuids: selectedRunUuids });
  }, [mutate, selectedRunUuids]);

  const columnPartitions = useMemo(
    () =>
      groupBy(
        Object.entries(selectedColumns),
        ([columnId]) =>
          parseEvalRunsTableKeyedColumnKey(columnId)?.columnType ?? EvalRunsTableKeyedColumnPrefix.ATTRIBUTE,
      ),
    [selectedColumns],
  );

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
        <Button
          css={{ flexShrink: 0 }}
          icon={<RefreshIcon />}
          disabled={isFetching}
          onClick={refetchRuns}
          componentId="mlflow.eval-runs.table-refresh-button"
        />
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
                        : parseEvalRunsTableKeyedColumnKey(column)?.key ?? column;

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
        {selectedRunUuids.length > 0 && (
          <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm }}>
            <Button danger componentId="select-all-runs-button" onClick={() => setDeleteModalVisible(true)}>
              <FormattedMessage defaultMessage="Delete" description="Delete runs" />
            </Button>
            <Modal
              componentId="mlflow.eval-runs.runs-delete-modal"
              visible={deleteModalVisible}
              onOk={handleDelete}
              okButtonProps={{ danger: true, loading: isLoading }}
              okText={
                <FormattedMessage defaultMessage="Delete" description="Delete evaluation runs modal button text" />
              }
              onCancel={() => {
                setDeleteModalVisible(false);
              }}
              cancelText={
                <FormattedMessage defaultMessage="Cancel" description="Delete evaluation runs cancel button text" />
              }
              confirmLoading={isLoading}
              title={
                <FormattedMessage
                  defaultMessage="Delete {numRuns, plural, =1 {1 run} other {# runs}}"
                  description="Delete evaluation runs modal title"
                  values={{ numRuns: selectedRunUuids.length }}
                />
              }
            >
              <FormattedMessage
                defaultMessage="Are you sure you want to delete these runs?"
                description="Delete evaluation runs modal confirmation text"
              />
            </Modal>
          </div>
        )}
      </div>
    </div>
  );
};
