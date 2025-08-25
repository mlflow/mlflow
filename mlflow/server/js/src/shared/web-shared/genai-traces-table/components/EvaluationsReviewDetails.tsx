import { useState } from 'react';

import { Alert, Button, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import type { UseQueryResult } from '@databricks/web-shared/query-client';

import { EvaluationsReviewAssessmentsSection } from './EvaluationsReviewAssessmentsSection';
import { EvaluationsReviewHeaderSection } from './EvaluationsReviewHeaderSection';
import { EvaluationsReviewInputSection } from './EvaluationsReviewInputSection';
import { EvaluationsReviewResponseSection } from './EvaluationsReviewResponseSection';
import { EvaluationsReviewRetrievalSection } from './EvaluationsReviewRetrievalSection';
import { getEvaluationResultTitle } from './GenAiEvaluationTracesReview.utils';
import { usePendingAssessmentEntries } from '../hooks/usePendingAssessmentEntries';
import type { AssessmentInfo, RunEvaluationResultAssessmentDraft, RunEvaluationTracesDataEntry } from '../types';

export const EvaluationsReviewDetailsHeader = ({
  evaluationResult,
}: {
  evaluationResult: RunEvaluationTracesDataEntry;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, overflow: 'hidden' }}>
      <Typography.Title
        level={2}
        withoutMargins
        css={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
      >
        {getEvaluationResultTitle(evaluationResult)}
      </Typography.Title>
    </div>
  );
};

export const EvaluationsReviewDetails = ({
  experimentId,
  evaluationResult,
  otherEvaluationResult,
  onSavePendingAssessments,
  onClickNext,
  runDisplayName,
  compareToRunDisplayName,
  isNextAvailable = false,
  isReadOnly = false,
  exportToEvalsInstanceEnabled = false,
  assessmentInfos,
  traceQueryResult,
  compareToTraceQueryResult,
}: {
  experimentId: string;
  evaluationResult: RunEvaluationTracesDataEntry;
  otherEvaluationResult?: RunEvaluationTracesDataEntry;
  onSavePendingAssessments?: (
    evaluationResult: RunEvaluationTracesDataEntry,
    pendingAssessments: RunEvaluationResultAssessmentDraft[],
  ) => Promise<void>;
  onClickNext?: () => void;
  runDisplayName?: string;
  compareToRunDisplayName?: string;
  isNextAvailable?: boolean;
  isReadOnly?: boolean;
  exportToEvalsInstanceEnabled?: boolean;
  assessmentInfos: AssessmentInfo[];
  traceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
  compareToTraceQueryResult: UseQueryResult<ModelTrace | undefined, unknown>;
}) => {
  const intl = useIntl();

  const { pendingAssessments, draftEvaluationResult, upsertAssessment, resetPendingAssessments } =
    usePendingAssessmentEntries(evaluationResult);

  const hasPendingAssessments = pendingAssessments.length > 0;

  // If user has already reviewed the evaluation, this flag allows to override the review and enable editing
  const [overridingExistingReview, setOverridingExistingReview] = useState(false);

  const hasErrorCode = Boolean(evaluationResult.errorCode);
  const hasErrorMessage = Boolean(evaluationResult.errorMessage);
  const showAlert = hasErrorCode || hasErrorMessage;
  const [alertExpanded, setAlertExpanded] = useState(false);
  const toggleAlertExpanded = () => setAlertExpanded((alertExpanded) => !alertExpanded);

  return (
    <>
      {showAlert ? (
        <>
          <Alert
            action={
              hasErrorMessage && (
                <Button
                  componentId={`mlflow.evaluations_review.evaluation_error_alert.show_${
                    alertExpanded ? 'less' : 'more'
                  }_button`}
                  onClick={toggleAlertExpanded}
                >
                  {alertExpanded ? (
                    <FormattedMessage defaultMessage="Show less" description="Button to close alert description" />
                  ) : (
                    <FormattedMessage defaultMessage="Show more" description="Button to expand alert description" />
                  )}
                </Button>
              )
            }
            closable={false}
            componentId="mlflow.evaluations_review.evaluation_error_alert"
            message={hasErrorCode ? `${evaluationResult.errorCode}` : 'UNKNOWN_ERROR'}
            description={alertExpanded && `${evaluationResult.errorMessage}`}
            type="error"
          />
          <Spacer size="md" />
        </>
      ) : null}
      <EvaluationsReviewHeaderSection
        experimentId={experimentId}
        runDisplayName={runDisplayName}
        otherRunDisplayName={compareToRunDisplayName}
        evaluationResult={evaluationResult}
        otherEvaluationResult={otherEvaluationResult}
        exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
        traceQueryResult={traceQueryResult}
        compareToTraceQueryResult={compareToTraceQueryResult}
      />
      <EvaluationsReviewAssessmentsSection
        evaluationResult={draftEvaluationResult}
        otherEvaluationResult={otherEvaluationResult}
        onUpsertAssessment={upsertAssessment}
        onClickNext={onClickNext}
        onSavePendingAssessments={async () => {
          // Save and reset the pending assessments
          await onSavePendingAssessments?.(evaluationResult, pendingAssessments);
          resetPendingAssessments();
        }}
        isNextAvailable={isNextAvailable}
        overridingExistingReview={overridingExistingReview}
        setOverridingExistingReview={setOverridingExistingReview}
        pendingAssessments={pendingAssessments}
        onResetPendingAssessments={resetPendingAssessments}
        isReadOnly={isReadOnly}
        assessmentInfos={assessmentInfos}
      />
      <EvaluationsReviewInputSection
        evaluationResult={evaluationResult}
        otherEvaluationResult={otherEvaluationResult}
      />
      <EvaluationsReviewResponseSection
        evaluationResult={evaluationResult}
        otherEvaluationResult={otherEvaluationResult}
      />
      <EvaluationsReviewRetrievalSection
        evaluationResult={draftEvaluationResult}
        otherEvaluationResult={otherEvaluationResult}
        onUpsertAssessment={upsertAssessment}
        overridingExistingReview={overridingExistingReview}
        isReadOnly={isReadOnly}
        assessmentInfos={assessmentInfos}
        traceQueryResult={traceQueryResult}
        compareToTraceQueryResult={compareToTraceQueryResult}
      />
      <Spacer size="lg" />
    </>
  );
};
