import { isNil } from 'lodash';
import React, { useCallback, useMemo } from 'react';

import {
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  GenericSkeleton,
  Modal,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ModelTraceExplorer, type ModelTrace } from '@databricks/web-shared/model-trace-explorer';

import { EvaluationsReviewDetailsHeader } from './EvaluationsReviewDetails';
import { GenAiEvaluationTracesReview } from './GenAiEvaluationTracesReview';
import { useGenAITracesTableConfig } from '../hooks/useGenAITracesTableConfig';
import { useGetTrace } from '../hooks/useGetTrace';
import type { AssessmentInfo, EvalTraceComparisonEntry, SaveAssessmentsQuery } from '../types';

const MODAL_SPACING_REM = 4;
const DEFAULT_MODAL_MARGIN_REM = 1;

export const GenAiEvaluationTracesReviewModal = React.memo(
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
    getTrace?: (traceId?: string) => Promise<ModelTrace | undefined>;
    saveAssessmentsQuery?: SaveAssessmentsQuery;
  }) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();

    const handleClose = useCallback(() => {
      onChangeEvaluationId(undefined);
    }, [onChangeEvaluationId]);

    // The URL always has an evaluation id, so we look in either current or other for the eval.
    const findEval = useCallback(
      (entry: EvalTraceComparisonEntry) =>
        entry.currentRunValue?.evaluationId === selectedEvaluationId ||
        entry.otherRunValue?.evaluationId === selectedEvaluationId,
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

    const traceQueryResult = useGetTrace(getTrace, evaluation?.currentRunValue?.traceInfo?.trace_id);
    const compareToTraceQueryResult = useGetTrace(getTrace, evaluation?.otherRunValue?.traceInfo?.trace_id);

    // Prefetching the next and previous traces to optimize performance
    useGetTrace(getTrace, nextEvaluation?.currentRunValue?.traceInfo?.trace_id);
    useGetTrace(getTrace, previousEvaluation?.currentRunValue?.traceInfo?.trace_id);

    // is true if only one of the two runs has a trace
    const isSingleTraceView = Boolean(evaluation?.currentRunValue) !== Boolean(evaluation?.otherRunValue);

    const currentTraceQueryResult =
      selectedEvaluationId === evaluation?.currentRunValue?.evaluationId ? traceQueryResult : compareToTraceQueryResult;

    if (isNil(evaluation)) {
      return <></>;
    }

    return (
      <div
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') {
            selectPreviousEval();
          } else if (e.key === 'ArrowRight') {
            selectNextEval();
          }
        }}
      >
        <Modal
          componentId="mlflow.evaluations_review.modal"
          visible
          title={
            evaluation.currentRunValue ? (
              <EvaluationsReviewDetailsHeader evaluationResult={evaluation.currentRunValue} />
            ) : evaluation.otherRunValue ? (
              <EvaluationsReviewDetailsHeader evaluationResult={evaluation.otherRunValue} />
            ) : null
          }
          onCancel={handleClose}
          size="wide"
          verticalSizing="maxed_out"
          css={{
            width: '100% !important',
            padding: `0 ${MODAL_SPACING_REM}rem !important`,
            [`& .${classNamePrefix}-modal-body`]: {
              flex: 1,
              paddingTop: 0,
            },
            [`& .${classNamePrefix}-modal-header`]: {
              paddingBottom: theme.spacing.sm,
            },
          }}
          footer={null} // Hide the footer
        >
          {currentTraceQueryResult.isFetching && (
            <GenericSkeleton
              label="Loading trace..."
              style={{
                // Size the width and height to fit the modal content area
                width: 'calc(100% - 45px)',
                height: 'calc(100% - 100px)',
                position: 'absolute',
                paddingRight: 500,
                zIndex: 2100,
                backgroundColor: theme.colors.backgroundPrimary,
              }}
            />
          )}
          {
            // Show ModelTraceExplorer only if there is no run to compare to and there's trace data.
            isSingleTraceView && !isNil(currentTraceQueryResult.data) ? (
              <div css={{ height: '100%', marginLeft: -theme.spacing.lg, marginRight: -theme.spacing.lg }}>
                <ModelTraceExplorerModalBody traceData={currentTraceQueryResult.data} />
              </div>
            ) : (
              evaluation.currentRunValue && (
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
              )
            )
          }
        </Modal>
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
            position: 'fixed',
            top: '50%',
            left: 0,
            zIndex: 2000,
            opacity: '.75',
            width: `${MODAL_SPACING_REM + DEFAULT_MODAL_MARGIN_REM}rem`,
            '&:hover': {
              opacity: '1.0',
            },
          }}
        >
          <div
            css={{
              backgroundColor: theme.colors.backgroundPrimary,
              borderRadius: theme.legacyBorders.borderRadiusMd,
              marginRight: theme.spacing.sm,
            }}
          >
            <Button
              disabled={!isPreviousAvailable}
              componentId="mlflow.evaluations_review.modal.previous_eval"
              icon={<ChevronLeftIcon />}
              onClick={() => selectPreviousEval()}
            />
          </div>
        </div>
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-start',
            position: 'fixed',
            top: '50%',
            right: 0,
            zIndex: 2000,
            width: `${MODAL_SPACING_REM + DEFAULT_MODAL_MARGIN_REM}rem`,
            opacity: '.75',
            '&:hover': {
              opacity: '1.0',
            },
          }}
        >
          <div
            css={{
              backgroundColor: theme.colors.backgroundPrimary,
              borderRadius: theme.legacyBorders.borderRadiusMd,
              marginLeft: theme.spacing.sm,
            }}
          >
            <Button
              disabled={!isNextAvailable}
              componentId="mlflow.evaluations_review.modal.next_eval"
              icon={<ChevronRightIcon />}
              onClick={() => selectNextEval()}
            />
          </div>
        </div>
      </div>
    );
  },
);

const ModelTraceExplorerModalBody = ({ traceData }: { traceData: ModelTrace }) => {
  return <ModelTraceExplorer modelTrace={traceData} />;
};
