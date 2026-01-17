import { isNil } from 'lodash';
import React, { useCallback, useMemo } from 'react';

import { GenericSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import {
  isV3ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerDrawer,
  ModelTraceExplorerSkeleton,
  isV4TraceId,
  type ModelTrace,
} from '@databricks/web-shared/model-trace-explorer';

import { EvaluationsReviewDetailsHeader } from './EvaluationsReviewDetails';
import { GenAiEvaluationTracesReview } from './GenAiEvaluationTracesReview';
import { useGenAITracesTableConfig } from '../hooks/useGenAITracesTableConfig';
import type { GetTraceFunction } from '../hooks/useGetTrace';
import { useGetTrace, useGetTraceByFullTraceId } from '../hooks/useGetTrace';
import type {
  AssessmentInfo,
  EvalTraceComparisonEntry,
  RunEvaluationTracesDataEntry,
  SaveAssessmentsQuery,
} from '../types';
import { convertTraceInfoV3ToRunEvalEntry, getSpansLocation, TRACKING_STORE_SPANS_LOCATION } from '../utils/TraceUtils';

const evalEntryMatchesEvaluationId = (evaluationId: string, entry?: RunEvaluationTracesDataEntry) => {
  if (isV4TraceId(evaluationId) && entry?.fullTraceId === evaluationId) {
    return true;
  }
  return entry?.evaluationId === evaluationId;
};

export const GenAiEvaluationTracesReviewModal = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    experimentId,
    runUuid,
    evaluations,
    selectedEvaluationId,
    onChangeEvaluationId,
    runDisplayName,
    otherRunDisplayName,
    exportToEvalsInstanceEnabled = false,
    assessmentInfos,
    getTrace,
    saveAssessmentsQuery,
  }: {
    experimentId: string;
    runUuid?: string;
    evaluations: EvalTraceComparisonEntry[];
    selectedEvaluationId: string;
    onChangeEvaluationId: (evaluationId: string | undefined) => void;
    runDisplayName?: string;
    otherRunDisplayName?: string;
    exportToEvalsInstanceEnabled?: boolean;
    assessmentInfos: AssessmentInfo[];
    getTrace?: GetTraceFunction;
    saveAssessmentsQuery?: SaveAssessmentsQuery;
  }) => {
    const { theme } = useDesignSystemTheme();

    const handleClose = useCallback(() => {
      onChangeEvaluationId(undefined);
    }, [onChangeEvaluationId]);

    const findEval = useCallback(
      (entry: EvalTraceComparisonEntry) =>
        evalEntryMatchesEvaluationId(selectedEvaluationId, entry.currentRunValue) ||
        evalEntryMatchesEvaluationId(selectedEvaluationId, entry.otherRunValue),
      [selectedEvaluationId],
    );

    const previousEvaluationIdx = useMemo(
      () => (evaluations ? evaluations?.findIndex(findEval) - 1 : undefined),
      [evaluations, findEval],
    );
    const isPreviousAvailable = useMemo(
      () => previousEvaluationIdx !== undefined && previousEvaluationIdx >= 0,
      [previousEvaluationIdx],
    );

    const nextEvaluationIdx = useMemo(
      () => (evaluations ? evaluations?.findIndex(findEval) + 1 : undefined),
      [evaluations, findEval],
    );
    const isNextAvailable = useMemo(
      () => nextEvaluationIdx !== undefined && nextEvaluationIdx < evaluations.length,
      [nextEvaluationIdx, evaluations],
    );

    const selectPreviousEval = useCallback(() => {
      if (evaluations === null || previousEvaluationIdx === undefined) return;

      const newEvalId =
        evaluations[previousEvaluationIdx]?.currentRunValue?.evaluationId ||
        evaluations[previousEvaluationIdx]?.otherRunValue?.evaluationId;
      onChangeEvaluationId(newEvalId);
    }, [evaluations, previousEvaluationIdx, onChangeEvaluationId]);

    const selectNextEval = useCallback(() => {
      if (evaluations === null || nextEvaluationIdx === undefined) return;

      const newEvalId =
        evaluations[nextEvaluationIdx]?.currentRunValue?.evaluationId ||
        evaluations[nextEvaluationIdx]?.otherRunValue?.evaluationId;
      onChangeEvaluationId(newEvalId);
    }, [evaluations, nextEvaluationIdx, onChangeEvaluationId]);

    const evaluation = useMemo(() => evaluations?.find(findEval), [evaluations, findEval]);
    const nextEvaluation = useMemo(
      () => (nextEvaluationIdx && evaluations ? evaluations?.[nextEvaluationIdx] : undefined),
      [evaluations, nextEvaluationIdx],
    );
    const previousEvaluation = useMemo(
      () => (previousEvaluationIdx && evaluations ? evaluations?.[previousEvaluationIdx] : undefined),
      [evaluations, previousEvaluationIdx],
    );

    const tracesTableConfig = useGenAITracesTableConfig();

    // Auto-polling until trace is complete if the backend supports returning partial spans
    const spansLocation = getSpansLocation(evaluation?.currentRunValue?.traceInfo);
    const shouldEnablePolling = spansLocation === TRACKING_STORE_SPANS_LOCATION;

    const traceQueryResult = useGetTrace(getTrace, evaluation?.currentRunValue?.traceInfo, shouldEnablePolling);
    const compareToTraceQueryResult = useGetTrace(getTrace, evaluation?.otherRunValue?.traceInfo, shouldEnablePolling);

    // In case that the selected evaluation is not provided upstream (but the list is loaded),
    // we lazily fetch the full trace data here
    const shouldFetchTraceBySearchParamId = useMemo(
      () => Boolean(evaluations) && !evaluation && Boolean(selectedEvaluationId),
      [evaluations, evaluation, selectedEvaluationId],
    );

    const traceBySearchParamQueryResult = useGetTraceByFullTraceId(
      getTrace,
      shouldFetchTraceBySearchParamId ? selectedEvaluationId : undefined,
    );

    // Prefetching the next and previous traces to optimize performance
    useGetTrace(getTrace, nextEvaluation?.currentRunValue?.traceInfo);
    useGetTrace(getTrace, previousEvaluation?.currentRunValue?.traceInfo);

    // True if only one of the two runs has a trace (single trace view vs comparison view)
    const isSingleTraceView = Boolean(evaluation?.currentRunValue) !== Boolean(evaluation?.otherRunValue);

    const currentTraceQueryResult = shouldFetchTraceBySearchParamId
      ? traceBySearchParamQueryResult
      : evalEntryMatchesEvaluationId(selectedEvaluationId, evaluation?.currentRunValue)
        ? traceQueryResult
        : compareToTraceQueryResult;

    if (isNil(evaluation) && !shouldFetchTraceBySearchParamId) {
      return <></>;
    }

    const renderModalTitle = () => {
      if (shouldFetchTraceBySearchParamId) {
        if (traceBySearchParamQueryResult.isLoading) {
          return (
            <GenericSkeleton
              css={{
                width: 200,
                height: theme.general.heightBase,
              }}
            />
          );
        }
        if (traceBySearchParamQueryResult.data?.info && isV3ModelTraceInfo(traceBySearchParamQueryResult.data?.info)) {
          const runEvalEntry = convertTraceInfoV3ToRunEvalEntry(traceBySearchParamQueryResult.data?.info);
          return <EvaluationsReviewDetailsHeader evaluationResult={runEvalEntry} />;
        }
      }
      return evaluation?.currentRunValue ? (
        <EvaluationsReviewDetailsHeader evaluationResult={evaluation.currentRunValue} />
      ) : evaluation?.otherRunValue ? (
        <EvaluationsReviewDetailsHeader evaluationResult={evaluation.otherRunValue} />
      ) : null;
    };

    const currentTraceInfo = evaluation?.currentRunValue?.traceInfo;

    const content = (
      <>
        {((shouldFetchTraceBySearchParamId && traceBySearchParamQueryResult?.data) || isSingleTraceView) &&
        !isNil(currentTraceQueryResult.data) ? (
          <div css={{ height: 'calc(100% - 34px)', marginLeft: -theme.spacing.lg, marginRight: -theme.spacing.lg }}>
            <ModelTraceExplorerModalBody traceData={currentTraceQueryResult.data} />
          </div>
        ) : (
          evaluation?.currentRunValue &&
          (currentTraceQueryResult.isFetching ? (
            <div css={{ marginLeft: -theme.spacing.lg, marginRight: -theme.spacing.lg }}>
              <ModelTraceExplorerSkeleton />
            </div>
          ) : (
            <div css={{ overflow: 'auto', height: '100%' }}>
              <GenAiEvaluationTracesReview
                experimentId={experimentId}
                evaluation={evaluation.currentRunValue}
                otherEvaluation={evaluation.otherRunValue}
                selectNextEval={selectNextEval}
                isNextAvailable={isNextAvailable}
                css={{ flex: 1, overflow: 'hidden' }}
                runUuid={runUuid}
                isReadOnly={!tracesTableConfig.enableRunEvaluationWriteFeatures}
                runDisplayName={runDisplayName}
                compareToRunDisplayName={otherRunDisplayName}
                exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
                assessmentInfos={assessmentInfos}
                traceQueryResult={traceQueryResult}
                compareToTraceQueryResult={compareToTraceQueryResult}
                saveAssessmentsQuery={saveAssessmentsQuery}
              />
            </div>
          ))
        )}
      </>
    );

    return (
      <ModelTraceExplorerDrawer
        handleClose={handleClose}
        isNextAvailable={isNextAvailable}
        isPreviousAvailable={isPreviousAvailable}
        selectNextEval={selectNextEval}
        selectPreviousEval={selectPreviousEval}
        renderModalTitle={renderModalTitle}
        isLoading={currentTraceQueryResult.isFetching}
        experimentId={experimentId}
        traceInfo={currentTraceInfo}
      >
        {content}
      </ModelTraceExplorerDrawer>
    );
  },
);

const ModelTraceExplorerModalBody = ({ traceData }: { traceData: ModelTrace }) => {
  return <ModelTraceExplorer modelTrace={traceData} />;
};
