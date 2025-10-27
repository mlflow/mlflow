import type { RowSelectionState } from '@tanstack/react-table';
import React, { useState, useMemo, useCallback } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import type { Assessment, ModelTrace, ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { AssessmentSchemaContextProvider } from '@databricks/web-shared/model-trace-explorer';

import { computeEvaluationsComparison } from './GenAiTracesTable.utils';
import { GenAiTracesTableBody } from './GenAiTracesTableBody';
import { useActiveEvaluation } from './hooks/useActiveEvaluation';
import { FilterOperator, TracesTableColumnGroup, TracesTableColumnType } from './types';
import type {
  AssessmentFilter,
  AssessmentInfo,
  TracesTableColumn,
  TraceInfoV3,
  EvaluationsOverviewTableSort,
  TableFilter,
} from './types';
import { sortAssessmentInfos } from './utils/AggregationUtils';
import { shouldEnableTagGrouping } from './utils/FeatureUtils';
import { applyTraceInfoV3ToEvalEntry } from './utils/TraceUtils';

interface GenAITracesTableBodyContainerProps {
  // Experiment metadata
  experimentId: string;
  currentRunDisplayName?: string;
  runUuid?: string;
  compareToRunUuid?: string;
  compareToRunDisplayName?: string;
  getRunColor?: (runUuid: string) => string;

  // Table metadata
  assessmentInfos: AssessmentInfo[];

  // Table data
  currentTraceInfoV3: TraceInfoV3[];
  compareToTraceInfoV3?: TraceInfoV3[];
  getTrace: (traceId?: string) => Promise<ModelTrace | undefined>;

  // Table state
  selectedColumns: TracesTableColumn[];
  allColumns: TracesTableColumn[];
  tableSort: EvaluationsOverviewTableSort | undefined;
  filters: TableFilter[];
  setFilters: (filters: TableFilter[]) => void;

  // TODO: Remove this in favor of unified tagging modal apis
  onTraceTagsEdit?: (trace: ModelTraceInfo) => void;

  // Configuration
  enableRowSelection?: boolean;
}

const GenAITracesTableBodyContainerImpl: React.FC<React.PropsWithChildren<GenAITracesTableBodyContainerProps>> =
  React.memo((props: GenAITracesTableBodyContainerProps) => {
    const {
      experimentId,
      currentTraceInfoV3,
      compareToTraceInfoV3,
      currentRunDisplayName,
      runUuid,
      compareToRunUuid,
      compareToRunDisplayName,
      setFilters,
      filters,
      selectedColumns,
      tableSort,
      assessmentInfos,
      getTrace,
      onTraceTagsEdit,
      allColumns,
      getRunColor,
      enableRowSelection = true,
    } = props;
    const { theme } = useDesignSystemTheme();

    // Convert trace info v3 to the format expected by GenAITracesTableBody
    const currentEvaluationResults = useMemo(
      () =>
        applyTraceInfoV3ToEvalEntry(
          currentTraceInfoV3.map((traceInfo) => ({
            evaluationId: traceInfo.trace_id,
            requestId: traceInfo.client_request_id || traceInfo.trace_id,
            inputsId: traceInfo.trace_id,
            inputs: {},
            outputs: {},
            targets: {},
            overallAssessments: [],
            responseAssessmentsByName: {},
            metrics: {},
            traceInfo,
          })),
        ),
      [currentTraceInfoV3],
    );
    const compareToEvaluationResults = useMemo(
      () =>
        applyTraceInfoV3ToEvalEntry(
          (compareToTraceInfoV3 || []).map((traceInfo) => ({
            evaluationId: traceInfo.trace_id,
            requestId: traceInfo.client_request_id || traceInfo.trace_id,
            inputsId: traceInfo.trace_id,
            inputs: {},
            outputs: {},
            targets: {},
            overallAssessments: [],
            responseAssessmentsByName: {},
            metrics: {},
            traceInfo,
          })),
        ),
      [compareToTraceInfoV3],
    );

    const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

    // Handle assessment filter toggle
    const handleAssessmentFilterToggle = useCallback(
      (assessmentName: string, filterValue: any, run: string) => {
        const filter = filters.find(
          (filter) => filter.column === TracesTableColumnGroup.ASSESSMENT && filter.key === assessmentName,
        );
        if (filter === undefined) {
          setFilters([
            ...filters,
            {
              column: TracesTableColumnGroup.ASSESSMENT,
              key: assessmentName,
              operator: FilterOperator.EQUALS,
              value: filterValue,
            },
          ]);
        } else if (filter.value === filterValue) {
          // Remove the filter because it already exists.
          setFilters(
            filters.filter(
              (filter) => !(filter.column === TracesTableColumnGroup.ASSESSMENT && filter.key === assessmentName),
            ),
          );
        } else {
          // Replace any filters with the same assessment name and run.
          setFilters(
            filters.map((filter) => {
              if (filter.column === TracesTableColumnGroup.ASSESSMENT && filter.key === assessmentName) {
                return {
                  column: TracesTableColumnGroup.ASSESSMENT,
                  key: assessmentName,
                  operator: FilterOperator.EQUALS,
                  value: filterValue,
                };
              }
              return filter;
            }),
          );
        }
      },
      [filters, setFilters],
    );

    const assessmentFilters: AssessmentFilter[] = useMemo(() => {
      return filters
        .filter((filter) => filter.column === TracesTableColumnGroup.ASSESSMENT)
        .map((filter) => ({
          assessmentName: filter.key || '',
          filterValue: filter.value,
          run: currentRunDisplayName || '',
        }));
    }, [filters, currentRunDisplayName]);

    const [selectedEvaluationId, setSelectedEvaluationId] = useActiveEvaluation();

    // Get selected assessment infos
    const selectedAssessmentInfos = useMemo(() => {
      const selectedAssessmentCols = selectedColumns.filter((col) => col.type === TracesTableColumnType.ASSESSMENT);
      const selectedAssessments = selectedAssessmentCols.map((col) => col.assessmentInfo as AssessmentInfo);
      return sortAssessmentInfos(selectedAssessments);
    }, [selectedColumns]);

    // Compute evaluations comparison
    const evaluationResults = useMemo(
      () => computeEvaluationsComparison(currentEvaluationResults, compareToEvaluationResults),
      [currentEvaluationResults, compareToEvaluationResults],
    );

    const assessments = useMemo(() => {
      return currentEvaluationResults.flatMap((evalResult) => evalResult?.traceInfo?.assessments ?? []) as Assessment[];
    }, [currentEvaluationResults]);

    return (
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
            gap: theme.spacing.md,
            width: '100%',
            flex: 1,
            overflowY: 'hidden',
          }}
        >
          <div
            css={{
              flex: 1,
              overflowY: 'hidden',
            }}
          >
            <AssessmentSchemaContextProvider assessments={assessments}>
              <GenAiTracesTableBody
                experimentId={experimentId}
                selectedColumns={selectedColumns}
                allColumns={allColumns}
                evaluations={evaluationResults}
                selectedEvaluationId={selectedEvaluationId}
                selectedAssessmentInfos={selectedAssessmentInfos}
                assessmentInfos={assessmentInfos}
                assessmentFilters={assessmentFilters}
                onChangeEvaluationId={setSelectedEvaluationId}
                getRunColor={getRunColor}
                runUuid={runUuid}
                compareToRunUuid={compareToRunUuid}
                runDisplayName={currentRunDisplayName}
                compareToRunDisplayName={compareToRunDisplayName}
                enableRowSelection={enableRowSelection}
                rowSelection={rowSelection}
                setRowSelection={setRowSelection}
                toggleAssessmentFilter={handleAssessmentFilterToggle}
                tableSort={tableSort}
                getTrace={getTrace}
                onTraceTagsEdit={onTraceTagsEdit}
                enableGrouping={shouldEnableTagGrouping()}
              />
            </AssessmentSchemaContextProvider>
          </div>
        </div>
      </div>
    );
  });

// TODO: Add an error boundary to the OSS trace table
export const GenAITracesTableBodyContainer = GenAITracesTableBodyContainerImpl;
