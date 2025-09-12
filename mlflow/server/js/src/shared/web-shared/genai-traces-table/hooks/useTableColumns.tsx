import { isNil } from 'lodash';
import { useMemo } from 'react';

import type { IntlShape } from '@databricks/i18n';

import { KnownEvaluationResultAssessmentName } from '../enum';
import type { AssessmentInfo, RunEvaluationTracesDataEntry, TraceInfoV3, TracesTableColumn } from '../types';
import { TracesTableColumnGroup, TracesTableColumnType } from '../types';
import { shouldEnableTagGrouping } from '../utils/FeatureUtils';
import {
  createCustomMetadataColumnId,
  createTagColumnId,
  MLFLOW_INTERNAL_PREFIX,
  shouldUseTraceInfoV3,
} from '../utils/TraceUtils';

export const USER_COLUMN_ID = 'user';
export const SESSION_COLUMN_ID = 'session';
export const RESPONSE_COLUMN_ID = 'response';
export const TRACE_ID_COLUMN_ID = 'trace_id';
export const REQUEST_TIME_COLUMN_ID = 'request_time';
export const EXECUTION_DURATION_COLUMN_ID = 'execution_duration';
export const STATE_COLUMN_ID = 'state';
export const SOURCE_COLUMN_ID = 'source';
export const TAGS_COLUMN_ID = 'tags';
export const TRACE_NAME_COLUMN_ID = 'trace_name';
export const INPUTS_COLUMN_ID = 'request';
export const RUN_NAME_COLUMN_ID = 'run_name';
export const LOGGED_MODEL_COLUMN_ID = 'logged_model';
export const TOKENS_COLUMN_ID = 'tokens';
export const CUSTOM_METADATA_COLUMN_ID = 'custom_metadata';

export const SORTABLE_INFO_COLUMNS = [EXECUTION_DURATION_COLUMN_ID, REQUEST_TIME_COLUMN_ID, SESSION_COLUMN_ID];
// Columns that are sortable by the server. Server-side sorting should be prioritized over client-side sorting.
export const SERVER_SORTABLE_INFO_COLUMNS = [EXECUTION_DURATION_COLUMN_ID, REQUEST_TIME_COLUMN_ID];

// This is a short term fix to not display any additional assessments with the trace info v3 migration.
// Long term we should decide how to best display these assessments.
const EXCLUDED_ASSESSMENT_NAMES = [
  KnownEvaluationResultAssessmentName.CHUNK_RELEVANCE,
  KnownEvaluationResultAssessmentName.TOTAL_INPUT_TOKEN_COUNT,
  KnownEvaluationResultAssessmentName.TOTAL_OUTPUT_TOKEN_COUNT,
  KnownEvaluationResultAssessmentName.TOTAL_TOKEN_COUNT,
  KnownEvaluationResultAssessmentName.DOCUMENT_RECALL,
  KnownEvaluationResultAssessmentName.DOCUMENT_RATINGS,
];

const ASSESSMENT_COLUMN_ID_SUFFIX = '_assessment_column';
const EXPECTATION_COLUMN_ID_SUFFIX = '_expectation_column';

// Add a suffix to the assessment name as the id to make it work for blank names.
export function createAssessmentColumnId(assessmentName: string) {
  return assessmentName + ASSESSMENT_COLUMN_ID_SUFFIX;
}

export function createExpectationColumnId(expectationName: string) {
  return expectationName + EXPECTATION_COLUMN_ID_SUFFIX;
}

export const useTableColumns = (
  intl: IntlShape,
  currentEvaluationResults: RunEvaluationTracesDataEntry[],
  assessmentInfos: AssessmentInfo[],
  runUuid: string | undefined,
  otherEvaluationResults?: RunEvaluationTracesDataEntry[],
  isTraceInfoV3Override?: boolean,
) => {
  const allColumns: TracesTableColumn[] = useMemo(() => {
    const isTraceInfoV3 = isTraceInfoV3Override ?? shouldUseTraceInfoV3(currentEvaluationResults);
    let inputCols = [];
    if (!isTraceInfoV3) {
      let inputKeys = new Set<string>();
      let traceInfoColumns = new Set<keyof TraceInfoV3>();

      currentEvaluationResults.forEach((result) => {
        const { inputs } = result;
        inputKeys = new Set<string>([...inputKeys, ...Object.keys(inputs || {})]);

        traceInfoColumns = new Set<keyof TraceInfoV3>([
          ...traceInfoColumns,
          ...Object.keys(result.traceInfo || {}),
        ] as (keyof TraceInfoV3)[]);
      });

      inputCols = [...inputKeys].map((key) => ({
        id: key,
        label: key,
        type: TracesTableColumnType.INPUT,
      }));
    } else {
      inputCols = [
        {
          id: INPUTS_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Request',
            description: 'Column label for request',
          }),
          type: TracesTableColumnType.INPUT,
          group: TracesTableColumnGroup.INFO,
        },
      ];
    }

    const assessmentColumns = assessmentInfos
      .map((assessmentInfo) => ({
        id: createAssessmentColumnId(assessmentInfo.name),
        label: assessmentInfo.displayName,
        type: TracesTableColumnType.ASSESSMENT,
        assessmentInfo,
        group: TracesTableColumnGroup.ASSESSMENT,
      }))
      .filter(
        (assessment) =>
          // retrieval columns should not be displayed in the table since they don't apply to the overall trace
          !assessment.assessmentInfo.isRetrievalAssessment &&
          !EXCLUDED_ASSESSMENT_NAMES.includes(assessment.assessmentInfo.name as KnownEvaluationResultAssessmentName),
      );

    let infoCols;
    const expectationColumns: Record<string, TracesTableColumn> = {};
    if (isTraceInfoV3) {
      infoCols = [
        {
          id: TRACE_ID_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Trace ID',
            description: 'Column label for trace ID',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: TRACE_NAME_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Trace name',
            description: 'Column label for trace name',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: RESPONSE_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Response',
            description: 'Column label for response',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: USER_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'User',
            description: 'Column label for user',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: SESSION_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Session',
            description: 'Column label for session',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: EXECUTION_DURATION_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Execution time',
            description: 'Column label for execution time',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: REQUEST_TIME_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Request time',
            description: 'Column label for request time',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: STATE_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'State',
            description: 'Column label for state',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: SOURCE_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Source',
            description: 'Column label for source',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: LOGGED_MODEL_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Version',
            description: 'Column label for logged model',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        {
          id: TOKENS_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Tokens',
            description: 'Column label for tokens',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        // Only show run name column on experiment level traces, where runUuid is not provided
        isNil(runUuid) && {
          id: RUN_NAME_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Run name',
            description: 'Column label for run name',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.INFO,
        },
        !shouldEnableTagGrouping() && {
          id: TAGS_COLUMN_ID,
          label: intl.formatMessage({
            defaultMessage: 'Tags',
            description: 'Column label for tags',
          }),
          type: TracesTableColumnType.TRACE_INFO,
          group: TracesTableColumnGroup.TAG,
        },
      ];

      const allResults = [...currentEvaluationResults, ...(otherEvaluationResults || [])];
      // Populate custom metadata columns
      const customMetadataColumns: Record<string, TracesTableColumn> = {};
      allResults.forEach((result: RunEvaluationTracesDataEntry) => {
        const traceMetadata = result.traceInfo?.trace_metadata;
        if (traceMetadata) {
          Object.keys(traceMetadata).forEach((key) => {
            if (!key.startsWith(MLFLOW_INTERNAL_PREFIX) && !customMetadataColumns[key]) {
              customMetadataColumns[key] = {
                id: createCustomMetadataColumnId(key),
                label: key,
                type: TracesTableColumnType.TRACE_INFO,
                group: TracesTableColumnGroup.INFO,
              };
            }
          });
        }

        const expectations = result.targets;
        if (expectations) {
          Object.keys(expectations).forEach((expectationName) => {
            if (!expectationColumns[expectationName]) {
              expectationColumns[expectationName] = {
                id: createExpectationColumnId(expectationName),
                label: expectationName,
                type: TracesTableColumnType.EXPECTATION,
                group: TracesTableColumnGroup.EXPECTATION,
                expectationName,
              };
            }
          });
        }
      });
      infoCols = [...infoCols, ...Object.values(customMetadataColumns)];

      if (shouldEnableTagGrouping()) {
        const tagColumnRecords: Record<string, TracesTableColumn> = {};
        allResults
          .map((result) => result.traceInfo?.tags)
          .forEach((tag) => {
            Object.keys(tag || {}).forEach((key) => {
              if (!key.startsWith(MLFLOW_INTERNAL_PREFIX) && !tagColumnRecords[key]) {
                tagColumnRecords[key] = {
                  id: createTagColumnId(key),
                  label: key,
                  type: TracesTableColumnType.TRACE_INFO,
                  group: TracesTableColumnGroup.TAG,
                };
              }
            });
          });
        const tagColumns = Object.values(tagColumnRecords);

        infoCols = [...infoCols, ...tagColumns];
      }
    } else {
      infoCols = currentEvaluationResults.some((result) => !isNil(result.requestTime))
        ? [
            {
              id: REQUEST_TIME_COLUMN_ID,
              label: intl.formatMessage({
                defaultMessage: 'Request time',
                description: 'Column label for request time',
              }),
              type: TracesTableColumnType.INTERNAL_MONITOR_REQUEST_TIME,
            },
          ]
        : [];
    }

    return [...inputCols, ...infoCols, ...assessmentColumns, ...Object.values(expectationColumns)].filter(
      (col): col is TracesTableColumn => Boolean(col),
    );
  }, [currentEvaluationResults, intl, assessmentInfos, runUuid, otherEvaluationResults, isTraceInfoV3Override]);

  return allColumns;
};
