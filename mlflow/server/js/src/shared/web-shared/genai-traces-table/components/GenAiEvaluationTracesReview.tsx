import { useCallback, useEffect, useRef } from 'react';

import { Spinner, useDesignSystemTheme } from '@databricks/design-system';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import type { UseQueryResult } from '@databricks/web-shared/query-client';

import { EvaluationsReviewDetails } from './EvaluationsReviewDetails';
import {
  copyAiOverallAssessmentAsHumanAssessment,
  shouldRepeatExistingOriginalOverallAiAssessment,
} from './GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, RunEvaluationTracesDataEntry, SaveAssessmentsQuery } from '../types';
import { RUN_EVALUATIONS_SINGLE_ITEM_REVIEW_UI_PAGE_ID } from '../utils/EvaluationLogging';

export const GenAiEvaluationTracesReview = ({
  experimentId,
  evaluation,
  otherEvaluation,
  className,
  runUuid,
  isReadOnly = false,
  selectNextEval,
  isNextAvailable,
  runDisplayName,
  compareToRunDisplayName,
  exportToEvalsInstanceEnabled = false,
  assessmentInfos,
  traceQueryResult,
  compareToTraceQueryResult,
  saveAssessmentsQuery,
}: {
  experimentId: string;
  evaluation: RunEvaluationTracesDataEntry;
  otherEvaluation?: RunEvaluationTracesDataEntry;
  className?: string;
  runUuid?: string;
  isReadOnly?: boolean;
  selectNextEval: () => void;
  isNextAvailable: boolean;
  runDisplayName?: string;
  compareToRunDisplayName?: string;
  exportToEvalsInstanceEnabled?: boolean;
  assessmentInfos: AssessmentInfo[];
  traceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
  compareToTraceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
  saveAssessmentsQuery?: SaveAssessmentsQuery;
}) => {
  const { theme } = useDesignSystemTheme();

  const handleSavePendingAssessments = useCallback(
    async (evaluation, pendingAssessments) => {
      if (!evaluation || isReadOnly || !runUuid || !saveAssessmentsQuery) {
        return;
      }

      // Prepare the list of assessments to be sent to the backend.
      const assessmentsToSave = pendingAssessments.slice();

      // Even if user did not provide any explicit overall assessment, we still should be able to mark the evaluation as reviewed.
      // We check if there are no user provided overall assessments and if the last overall assessment was AI generated.
      const shouldRepeatOverallAiAssessment = shouldRepeatExistingOriginalOverallAiAssessment(
        evaluation,
        pendingAssessments,
      );

      // If we should repeat the AI generated overall assessment, we need to copy and add it to the list of assessments to save.
      if (shouldRepeatOverallAiAssessment) {
        const repeatedOverallAssessment = copyAiOverallAssessmentAsHumanAssessment(evaluation);

        repeatedOverallAssessment && assessmentsToSave.unshift(repeatedOverallAssessment);
      }

      return saveAssessmentsQuery.savePendingAssessments(runUuid, evaluation.evaluationId, assessmentsToSave);
    },
    [runUuid, isReadOnly, saveAssessmentsQuery],
  );

  // Scroll right side panel to the top when switching between evaluations
  const reviewDetailsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    reviewDetailsRef.current?.scrollTo(0, 0);
  }, [evaluation.evaluationId]);

  return (
    <div
      // comment for copybara formatting
      css={{ display: 'flex', position: 'relative', overflow: 'scroll' }}
      className={className}
    >
      <div
        css={{
          flex: 1,
          overflow: 'auto',
        }}
        ref={reviewDetailsRef}
      >
        <EvaluationsReviewDetails
          experimentId={experimentId}
          key={evaluation.evaluationId}
          evaluationResult={evaluation}
          otherEvaluationResult={otherEvaluation}
          onSavePendingAssessments={handleSavePendingAssessments}
          onClickNext={selectNextEval}
          isNextAvailable={isNextAvailable}
          isReadOnly={isReadOnly || !runUuid}
          runDisplayName={runDisplayName}
          compareToRunDisplayName={compareToRunDisplayName}
          exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
          assessmentInfos={assessmentInfos}
          traceQueryResult={traceQueryResult}
          compareToTraceQueryResult={compareToTraceQueryResult}
        />
      </div>
      {saveAssessmentsQuery?.isSaving && (
        <div
          css={{
            inset: 0,
            position: 'absolute',
            backgroundColor: theme.colors.overlayOverlay,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: theme.options.zIndexBase + 1,
          }}
        >
          <Spinner size="large" inheritColor css={{ color: theme.colors.backgroundPrimary }} />
        </div>
      )}
    </div>
  );
};
