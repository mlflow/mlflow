import type { RowSelectionState } from '@tanstack/react-table';
import { isNil } from 'lodash';
import React, { useCallback, useMemo, useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  Typography,
  useDesignSystemTheme,
  DialogCombobox,
  DialogComboboxContent,
  SegmentedControlGroup,
  SegmentedControlButton,
  DialogComboboxCustomButtonTriggerWrapper,
  XCircleFillIcon,
  TableFilterLayout,
  LegacySkeleton,
  FilterIcon,
  Tooltip,
  Spinner,
  WarningIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { ModelTrace, ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';

import { GenAITracesTableActions } from './GenAITracesTableActions';
import { computeEvaluationsComparison } from './GenAiTracesTable.utils';
import { GenAiTracesTableBody } from './GenAiTracesTableBody';
import { GenAiTracesTableSearchInput } from './GenAiTracesTableSearchInput';
import { EvaluationsOverviewColumnSelector } from './components/EvaluationsOverviewColumnSelector';
import { EvaluationsOverviewSortDropdown } from './components/EvaluationsOverviewSortDropdown';
import { GenAiEvaluationBadge } from './components/GenAiEvaluationBadge';
import {
  getAssessmentValueLabel,
  KnownEvaluationResultAssessmentName,
} from './components/GenAiEvaluationTracesReview.utils';
import { useActiveEvaluation } from './hooks/useActiveEvaluation';
import {
  assessmentValueToSerializedString,
  serializedStringToAssessmentValue,
  useAssessmentFilters,
} from './hooks/useAssessmentFilters';
import { useEvaluationsSearchQuery } from './hooks/useEvaluationsSearchQuery';
import { GenAITracesTableConfigProvider, type GenAITracesTableConfig } from './hooks/useGenAITracesTableConfig';
import { useTableColumns } from './hooks/useTableColumns';
import { TracesTableColumnType } from './types';
import type {
  AssessmentFilter,
  AssessmentInfo,
  AssessmentValueType,
  SaveAssessmentsQuery,
  RunEvaluationTracesDataEntry,
  TracesTableColumn,
  TraceActions,
  EvaluationsOverviewTableSort,
} from './types';
import { getAssessmentInfos, sortAssessmentInfos } from './utils/AggregationUtils';
import { displayPercentage } from './utils/DisplayUtils';
import { FILTER_DROPDOWN_COMPONENT_ID } from './utils/EvaluationLogging';
import { filterEvaluationResults } from './utils/EvaluationsFilterUtils';
import { applyTraceInfoV3ToEvalEntry } from './utils/TraceUtils';

function GenAiTracesTableImpl({
  experimentId,
  currentEvaluationResults: oldEvalResults,
  compareToEvaluationResults: oldCompareToEvalResults,
  currentRunDisplayName,
  runUuid,
  compareToRunUuid,
  compareToRunDisplayName,
  compareToRunLoading,
  sampledInfo,
  exportToEvalsInstanceEnabled = false,
  getTrace,
  saveAssessmentsQuery,
  enableRunEvaluationWriteFeatures,
  defaultSortOption,
  disableAssessmentTooltips,
  onTraceTagsEdit,
  traceActions,
  initialSelectedColumns,
}: {
  experimentId: string;
  currentEvaluationResults: RunEvaluationTracesDataEntry[];
  compareToEvaluationResults?: RunEvaluationTracesDataEntry[];
  currentRunDisplayName?: string;
  runUuid?: string;
  compareToRunUuid?: string;
  compareToRunDisplayName?: string;
  compareToRunLoading?: boolean;
  sampledInfo?: SampleInfo;
  exportToEvalsInstanceEnabled?: boolean;
  getTrace?: (traceId?: string) => Promise<ModelTrace | undefined>;
  saveAssessmentsQuery?: SaveAssessmentsQuery;
  enableRunEvaluationWriteFeatures?: boolean;
  defaultSortOption?: EvaluationsOverviewTableSort;
  // This is a temporary hack to disable assessment hovercards because
  // we don't properly display long strings yet. We should eventually fix the hovercard
  // to display long strings and remove this prop.
  disableAssessmentTooltips?: boolean;
  onTraceTagsEdit?: (trace: ModelTraceInfo) => void;
  traceActions?: TraceActions;
  initialSelectedColumns?: (allColumns: TracesTableColumn[]) => TracesTableColumn[];
}) {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Convert trace info v3 back to the old RunEvaluationTracesDataEntry format.
  // We should gradually migrate this component to use trace info v3 directly and
  // then we can remove this conversion.
  const currentEvaluationResults = applyTraceInfoV3ToEvalEntry(oldEvalResults);
  const compareToEvaluationResults = applyTraceInfoV3ToEvalEntry(oldCompareToEvalResults || []);

  const enableTableRowSelection: boolean = Object.keys(traceActions ?? {}).length > 0;

  const [selectedEvaluationId, setSelectedEvaluationId] = useActiveEvaluation();

  // evaluationResults contains the merged evaluation results from the current and compare-to runs.
  const evaluationResults = useMemo(
    () => computeEvaluationsComparison(currentEvaluationResults, compareToEvaluationResults),
    [currentEvaluationResults, compareToEvaluationResults],
  );

  const [searchQuery, setSearchQuery] = useEvaluationsSearchQuery();

  const assessmentInfos = useMemo(() => {
    return getAssessmentInfos(intl, currentEvaluationResults, compareToEvaluationResults);
  }, [intl, currentEvaluationResults, compareToEvaluationResults]);

  const allColumns: TracesTableColumn[] = useTableColumns(intl, currentEvaluationResults, assessmentInfos, runUuid);

  const [assessmentFilters, setAssessmentFilters] = useAssessmentFilters(
    // These are the subset of all assessment infos that are shown in the table
    allColumns.map((col) => col.assessmentInfo).filter((info): info is AssessmentInfo => info !== undefined),
  );

  const displayEvaluationResults = useMemo(() => {
    if (searchQuery === '' && assessmentFilters.length === 0) {
      return evaluationResults;
    }
    return filterEvaluationResults(
      evaluationResults,
      assessmentFilters,
      searchQuery,
      currentRunDisplayName,
      compareToRunDisplayName,
    );
  }, [evaluationResults, searchQuery, assessmentFilters, currentRunDisplayName, compareToRunDisplayName]);

  // TODO(nsthorat): Add these to the URL.
  // Initially all assessments, inputs, and certain info columns are selected
  const [selectedColumns, setSelectedColumns] = useState<TracesTableColumn[]>(allColumns);

  const selectedAssessmentInfos = useMemo(() => {
    const selectedAssessmentCols = selectedColumns.filter((col) => col.type === TracesTableColumnType.ASSESSMENT);
    const selectedAssessments = selectedAssessmentCols.map((col) => col.assessmentInfo as AssessmentInfo);
    const sortedSelectedAssessments = sortAssessmentInfos(selectedAssessments);
    return sortedSelectedAssessments;
  }, [selectedColumns]);

  const overallAssessmentCol = allColumns.find(
    (col) =>
      col.type === TracesTableColumnType.ASSESSMENT &&
      col.id === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
  );

  const initialSort: EvaluationsOverviewTableSort | undefined =
    defaultSortOption ||
    (overallAssessmentCol
      ? { key: overallAssessmentCol.id, type: TracesTableColumnType.ASSESSMENT, asc: true }
      : undefined);
  const [tableSort, setTableSort] = useState<EvaluationsOverviewTableSort | undefined>(initialSort);

  const getAssessmentFilter = useCallback(
    (assessmentName: string, run: string): AssessmentFilter | undefined => {
      return assessmentFilters.find((filter) => filter.assessmentName === assessmentName && filter.run === run);
    },
    [assessmentFilters],
  );
  const removeAssessmentFilter = useCallback(
    (assessmentName: string, run: string) => {
      setAssessmentFilters(
        assessmentFilters.filter((filter) => filter.assessmentName !== assessmentName || filter.run !== run),
      );
    },
    [assessmentFilters, setAssessmentFilters],
  );

  const updateAssessmentFilter = useCallback(
    (
      assessmentName: string,
      filterValue: AssessmentValueType,
      run: string,
      filterType?: AssessmentFilter['filterType'],
    ) => {
      const filter = assessmentFilters.find((filter) => filter.assessmentName === assessmentName && filter.run === run);
      if (filter === undefined) {
        setAssessmentFilters([
          ...assessmentFilters,
          {
            assessmentName,
            filterValue,
            filterType,
            run,
          },
        ]);
      } else {
        setAssessmentFilters(
          assessmentFilters.map((filter) => {
            if (filter.assessmentName === assessmentName) {
              return {
                ...filter,
                filterValue,
                filterType,
              };
            }
            return filter;
          }),
        );
      }
    },
    [assessmentFilters, setAssessmentFilters],
  );
  const toggleAssessmentFilter = useCallback(
    (
      assessmentName: string,
      filterValue: AssessmentValueType,
      run: string,
      filterType?: AssessmentFilter['filterType'],
    ) => {
      const filter = assessmentFilters.find((filter) => filter.assessmentName === assessmentName && filter.run === run);
      if (filter === undefined) {
        setAssessmentFilters([
          ...assessmentFilters,
          {
            assessmentName,
            filterValue,
            filterType,
            run,
          },
        ]);
      } else if (filter.filterValue === filterValue && filter.filterType === filterType) {
        // Remove the filter because it already exists.
        setAssessmentFilters(
          assessmentFilters.filter((filter) => filter.assessmentName !== assessmentName || filter.run !== run),
        );
      } else {
        // Replace any filters with the same assessment name and run.
        setAssessmentFilters(
          assessmentFilters.map((filter) => {
            if (filter.assessmentName === assessmentName && filter.run === run) {
              return {
                assessmentName,
                filterValue,
                filterType,
                run,
              };
            }
            return filter;
          }),
        );
      }
    },
    [assessmentFilters, setAssessmentFilters],
  );
  const clearFilters = useCallback(() => {
    setAssessmentFilters([]);
  }, [setAssessmentFilters]);

  const hasActiveFilters = assessmentFilters.length > 0;

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const selectedTraces: RunEvaluationTracesDataEntry[] = useMemo(() => {
    const selectedEvaluationIds = Object.keys(rowSelection).filter((evaluationId) => rowSelection[evaluationId]);
    return displayEvaluationResults
      .filter(
        (evaluation) =>
          evaluation.currentRunValue && selectedEvaluationIds.includes(evaluation.currentRunValue.evaluationId),
      )
      .map((entry) => entry.currentRunValue)
      .filter((entry) => entry !== undefined);
  }, [rowSelection, displayEvaluationResults]);

  const config: Partial<GenAITracesTableConfig> = {
    enableRunEvaluationWriteFeatures: enableRunEvaluationWriteFeatures,
  };

  const totalEvaluationResults = currentEvaluationResults.length;

  if (compareToRunLoading) {
    return <LegacySkeleton />;
  }

  return (
    <GenAITracesTableConfigProvider config={config}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.lg,
          overflow: 'hidden',
          height: '100%',
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            overflow: 'hidden',
            flexGrow: 1,
          }}
        >
          <div
            css={{
              display: 'flex',
              flex: 1,
              flexDirection: 'column',
              gap: theme.spacing.xs,
              overflow: 'hidden',
            }}
          >
            <div
              css={{
                display: 'flex',
                width: '100%',
                alignItems: 'flex-end',
                justifyContent: 'space-between',
                padding: `${theme.spacing.xs}px 0px`,
              }}
            >
              <TableFilterLayout
                css={{
                  marginBottom: 0,
                }}
              >
                <GenAiTracesTableSearchInput searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
                <DialogCombobox
                  componentId={FILTER_DROPDOWN_COMPONENT_ID}
                  label="Filters"
                  value={Array.from(assessmentFilters).map((filter) => filter.assessmentName)}
                  multiSelect
                >
                  <DialogComboboxCustomButtonTriggerWrapper>
                    <Button
                      endIcon={<ChevronDownIcon />}
                      componentId="mlflow.evaluations_review.table_ui.filter_button"
                      css={{
                        border: hasActiveFilters ? `1px solid ${theme.colors.actionDefaultBorderFocus} !important` : '',
                        backgroundColor: hasActiveFilters
                          ? `${theme.colors.actionDefaultBackgroundHover} !important`
                          : '',
                      }}
                    >
                      <div
                        css={{
                          display: 'flex',
                          gap: theme.spacing.sm,
                          alignItems: 'center',
                        }}
                      >
                        <FilterIcon />
                        {intl.formatMessage(
                          {
                            defaultMessage: 'Filters{numFilters}',
                            description: 'Evaluation review > evaluations list > filter dropdown button',
                          },
                          {
                            numFilters: hasActiveFilters ? ` (${assessmentFilters.length})` : '',
                          },
                        )}
                        {assessmentFilters.length > 0 && (
                          <XCircleFillIcon
                            css={{
                              fontSize: 12,
                              cursor: 'pointer',
                              color: theme.colors.grey400,
                              '&:hover': {
                                color: theme.colors.grey600,
                              },
                            }}
                            onClick={(e) => {
                              clearFilters();
                              e.stopPropagation();
                              e.preventDefault();
                            }}
                          />
                        )}
                      </div>
                    </Button>
                  </DialogComboboxCustomButtonTriggerWrapper>
                  <DialogComboboxContent>
                    <div
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
                        gap: theme.spacing.md,
                      }}
                    >
                      {selectedAssessmentInfos
                        // For now, we don't support filtering on numeric values.
                        .filter((info) => info.dtype !== 'numeric')
                        .map((assessmentInfo) => (
                          <div
                            css={{
                              display: 'flex',
                              flexDirection: 'column',
                              gap: theme.spacing.sm,
                            }}
                            key={assessmentInfo.name}
                          >
                            <div
                              css={{
                                fontWeight: 500,
                              }}
                            >
                              {assessmentInfo.displayName}
                            </div>
                            {currentRunDisplayName && (
                              <AssessmentsFilterSelector
                                assessmentLabel={assessmentInfo.displayName}
                                assessmentName={assessmentInfo.name}
                                assessmentInfo={assessmentInfo}
                                assessmentFilter={getAssessmentFilter(assessmentInfo.name, currentRunDisplayName)}
                                updateAssessmentFilter={updateAssessmentFilter}
                                removeAssessmentFilter={removeAssessmentFilter}
                                run={currentRunDisplayName}
                              />
                            )}
                            {compareToRunUuid && compareToRunDisplayName && (
                              <div
                                css={{
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: theme.spacing.xs,
                                }}
                              >
                                <Typography.Hint>{compareToRunDisplayName}</Typography.Hint>
                                <AssessmentsFilterSelector
                                  assessmentLabel={assessmentInfo.displayName}
                                  assessmentName={assessmentInfo.name}
                                  assessmentInfo={assessmentInfo}
                                  assessmentFilter={getAssessmentFilter(assessmentInfo.name, compareToRunDisplayName)}
                                  updateAssessmentFilter={updateAssessmentFilter}
                                  removeAssessmentFilter={removeAssessmentFilter}
                                  run={compareToRunDisplayName}
                                />
                              </div>
                            )}
                          </div>
                        ))}
                    </div>
                  </DialogComboboxContent>
                </DialogCombobox>
                <EvaluationsOverviewSortDropdown
                  tableSort={tableSort}
                  columns={allColumns}
                  onChange={(sortOption, orderByAsc) => {
                    setTableSort({ key: sortOption.key, type: sortOption.type, asc: orderByAsc });
                  }}
                />

                {/* Column selector */}
                <EvaluationsOverviewColumnSelector
                  columns={allColumns}
                  selectedColumns={selectedColumns}
                  setSelectedColumns={setSelectedColumns}
                />
                <GenAITracesTableActions
                  experimentId={experimentId}
                  selectedTraces={selectedTraces}
                  setRowSelection={setRowSelection}
                  traceActions={traceActions}
                  traceInfos={undefined}
                />
              </TableFilterLayout>
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                }}
              >
                <SampledInfoBadge totalRowCount={totalEvaluationResults} sampledInfo={sampledInfo} />
                <Typography.Hint>
                  {intl.formatMessage(
                    {
                      defaultMessage: '{numFilteredEvals} of {numEvals}',
                      description:
                        'Text displayed when showing a filtered subset evaluations in the evaluation review page',
                    },
                    {
                      numEvals: evaluationResults.length,
                      numFilteredEvals: displayEvaluationResults.length,
                    },
                  )}
                </Typography.Hint>
              </div>
            </div>
            <div
              css={{
                display: 'flex',
                gap: theme.spacing.md,
                width: '100%',
                flex: 1,
                overflowY: 'hidden',
              }}
            >
              <div
                css={{
                  display: 'flex',
                  flex: 1,
                  overflowY: 'hidden',
                }}
              >
                <GenAiTracesTableBody
                  experimentId={experimentId}
                  selectedColumns={selectedColumns}
                  allColumns={allColumns}
                  evaluations={displayEvaluationResults}
                  selectedEvaluationId={selectedEvaluationId}
                  selectedAssessmentInfos={selectedAssessmentInfos}
                  assessmentInfos={assessmentInfos}
                  assessmentFilters={assessmentFilters}
                  onChangeEvaluationId={setSelectedEvaluationId}
                  tableSort={tableSort}
                  runUuid={runUuid}
                  compareToRunUuid={compareToRunUuid}
                  runDisplayName={currentRunDisplayName}
                  compareToRunDisplayName={compareToRunDisplayName}
                  enableRowSelection={enableTableRowSelection}
                  rowSelection={enableTableRowSelection ? rowSelection : undefined}
                  setRowSelection={enableTableRowSelection ? setRowSelection : undefined}
                  exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
                  getTrace={getTrace}
                  toggleAssessmentFilter={toggleAssessmentFilter}
                  saveAssessmentsQuery={saveAssessmentsQuery}
                  disableAssessmentTooltips={disableAssessmentTooltips}
                  onTraceTagsEdit={onTraceTagsEdit}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </GenAITracesTableConfigProvider>
  );
}

const SampledInfoBadge = (props: { totalRowCount: number; sampledInfo?: SampleInfo }) => {
  const { totalRowCount, sampledInfo } = props;
  const intl = useIntl();

  if (!sampledInfo) {
    return null;
  }

  // TODO: Remove this once table supports pagination
  if (sampledInfo.maxAllowedCount) {
    if (totalRowCount >= sampledInfo.maxAllowedCount) {
      return (
        <Tooltip
          componentId="mlflow.experiment_list_view.max_traces.tooltip"
          content={intl.formatMessage(
            {
              defaultMessage: 'Only the top {evalResultsCount} results are shown',
              description: 'Evaluation review > evaluations list > sample info tooltip',
            },
            {
              evalResultsCount: totalRowCount,
            },
          )}
        >
          <WarningIcon color="warning" />
        </Tooltip>
      );
    }

    return null;
  }
  if (sampledInfo.logCountLoading || isNil(sampledInfo.sampledCount) || isNil(sampledInfo.totalCount)) {
    return <Spinner />;
  }
  return (
    <GenAiEvaluationBadge>
      <Tooltip
        componentId="mlflow.experiment_list_view.sampled_badge.tooltip"
        content={intl.formatMessage(
          {
            defaultMessage: 'Retrieved {sampledCount} out of {totalCount} total logs ({percentage}%)',
            description: 'Evaluation review > evaluations list > sample info tooltip',
          },
          {
            sampledCount: sampledInfo.sampledCount,
            totalCount: sampledInfo.totalCount,
            percentage: displayPercentage(sampledInfo.sampledCount / sampledInfo.totalCount),
          },
        )}
      >
        Sampled {displayPercentage(sampledInfo.sampledCount / sampledInfo.totalCount)}%
      </Tooltip>
    </GenAiEvaluationBadge>
  );
};

const ANY_VALUE = '__any_value__';

const AssessmentsFilterSelector = React.memo(
  ({
    assessmentName,
    assessmentInfo,
    assessmentLabel,
    assessmentFilter,
    updateAssessmentFilter,
    removeAssessmentFilter,
    run,
  }: {
    assessmentName: string;
    assessmentInfo: AssessmentInfo;
    assessmentLabel: string;
    assessmentFilter: AssessmentFilter | undefined;
    updateAssessmentFilter: (assessmentName: string, filterValue: AssessmentValueType, run: string) => void;
    removeAssessmentFilter: (assessmentName: string, run: string) => void;
    run: string;
  }) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();

    const selectedValue = assessmentFilter
      ? assessmentValueToSerializedString(assessmentFilter.filterValue)
      : ANY_VALUE;

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <SegmentedControlGroup
          name="size-story"
          value={selectedValue}
          onChange={(event) => {
            if (event.target.value === ANY_VALUE) {
              removeAssessmentFilter(assessmentName, run);
              return;
            }
            const value = serializedStringToAssessmentValue(assessmentInfo, event.target.value);
            updateAssessmentFilter(assessmentName, value, run);
          }}
          size="middle"
          componentId={`mlflow.evaluations_review.table_ui.filter_control_${assessmentName}`}
        >
          <SegmentedControlButton value={ANY_VALUE}>
            <div
              css={{
                display: 'flex',
                gap: theme.spacing.sm,
                alignItems: 'center',
                color: theme.colors.textPrimary,
              }}
            >
              {intl.formatMessage({
                defaultMessage: 'All',
                description:
                  'Evaluation review > sidebar list of evaluation results > filter control > option for filtering evaluation results with no filter',
              })}
            </div>
          </SegmentedControlButton>
          {Array.from(assessmentInfo.uniqueValues.values()).map((value) => {
            const { content, icon } = getAssessmentValueLabel(intl, theme, assessmentInfo, value);
            return (
              <SegmentedControlButton
                value={assessmentValueToSerializedString(value)}
                key={assessmentValueToSerializedString(value)}
              >
                <div
                  css={{
                    display: 'flex',
                    gap: theme.spacing.sm,
                    alignItems: 'center',
                    color: theme.colors.textPrimary,
                  }}
                >
                  {content}
                  {icon}
                </div>
              </SegmentedControlButton>
            );
          })}
        </SegmentedControlGroup>
      </div>
    );
  },
);

export interface SampleInfo {
  sampledCount?: number;
  totalCount?: number;
  logCountLoading: boolean;
  // Set this to the max number of logs that our system can fetch.
  // If the evaluation result count hits this number, we will show a warning indicator to the user.
  maxAllowedCount?: number;
}

// TODO: Add an error boundary to the OSS trace table
export const GenAiTracesTable = GenAiTracesTableImpl;
