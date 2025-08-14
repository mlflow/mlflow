import { first } from 'lodash';

import {
  Button,
  CheckCircleIcon,
  PencilIcon,
  Spacer,
  SparkleDoubleIcon,
  Typography,
  useDesignSystemTheme,
  Tooltip,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { EvaluationsReviewAssessmentTag } from './EvaluationsReviewAssessmentTag';
import { EvaluationsReviewAssessments } from './EvaluationsReviewAssessments';
import { EvaluationsReviewAssessmentsConfirmButton } from './EvaluationsReviewAssessmentsConfirmButton';
import {
  KnownEvaluationResultAssessmentName,
  getOrderedAssessments,
  isEvaluationResultReviewedAlready,
  KnownEvaluationResponseAssessmentNames,
  isAssessmentMissing,
} from './GenAiEvaluationTracesReview.utils';
import { VerticalBar } from './VerticalBar';
import type {
  AssessmentInfo,
  RunEvaluationResultAssessment,
  RunEvaluationResultAssessmentDraft,
  RunEvaluationTracesDataEntry,
} from '../types';

/**
 * Displays section with a list of evaluation assessments: overall and detailed.
 */
const EvaluationsReviewSingleRunAssessmentsSection = ({
  evaluationResult,
  onUpsertAssessment,
  onSavePendingAssessments,
  onClickNext,
  onResetPendingAssessments,
  isNextAvailable,
  overridingExistingReview = false,
  pendingAssessments = [],
  setOverridingExistingReview,
  isReadOnly = false,
  assessmentInfos,
}: {
  evaluationResult: RunEvaluationTracesDataEntry;
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  onSavePendingAssessments: () => Promise<void>;
  onClickNext?: () => void;
  onResetPendingAssessments?: () => void;
  isNextAvailable?: boolean;
  overridingExistingReview?: boolean;
  pendingAssessments?: RunEvaluationResultAssessmentDraft[];
  setOverridingExistingReview: (override: boolean) => void;
  isReadOnly?: boolean;
  assessmentInfos: AssessmentInfo[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (!evaluationResult) {
    return null;
  }
  const currentOverallAssessment = first(evaluationResult.overallAssessments);
  const rootCauseAssessmentName = currentOverallAssessment?.rootCauseAssessment?.assessmentName;
  const rootCauseAssessment = rootCauseAssessmentName
    ? first(evaluationResult?.responseAssessmentsByName[rootCauseAssessmentName])
    : undefined;

  const toBeReviewed =
    !isReadOnly && (!isEvaluationResultReviewedAlready(evaluationResult) || overridingExistingReview);

  const reopenReviewTooltip = intl.formatMessage({
    defaultMessage: 'Reopen review',
    description: 'Evaluation review > assessments > reopen review tooltip',
  });

  const overallAssessmentsByName: [string, RunEvaluationResultAssessment[]][] = [
    [KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT, evaluationResult.overallAssessments],
  ];

  const overallAssessmentInfo = assessmentInfos.find(
    (info) => info.name === KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
  );

  return (
    <div css={{ width: '100%' }}>
      <div css={{ width: '100%', paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md }}>
        <div
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            padding: theme.spacing.md,
            paddingBottom: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {overallAssessmentInfo && (
            <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
              <SparkleDoubleIcon color="ai" />
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Overall assessment:"
                  description="Evaluation review > assessments > overall assessment > title"
                />
              </Typography.Text>
              {/* TODO: make overall assessment editable */}
              <EvaluationsReviewAssessmentTag
                assessment={currentOverallAssessment}
                aria-label={KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT}
                disableJudgeTypeIcon={isAssessmentMissing(currentOverallAssessment)}
                assessmentInfo={overallAssessmentInfo}
                type="assessment-value"
              />
            </div>
          )}
          <EvaluationsReviewAssessments
            assessmentsType="overall"
            assessmentsByName={overallAssessmentsByName}
            rootCauseAssessment={rootCauseAssessment}
            onUpsertAssessment={onUpsertAssessment}
            allowEditing={toBeReviewed}
            alwaysExpanded
            options={[KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT]}
            assessmentInfos={assessmentInfos}
          />
          <div
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.general.borderRadiusBase,
              padding: theme.spacing.sm,
            }}
          >
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Detailed assessments"
                description="Evaluation review > assessments > detailed assessments > title"
              />
            </Typography.Text>
            <Spacer size="sm" />
            <EvaluationsReviewAssessments
              assessmentsType="response"
              assessmentsByName={getOrderedAssessments(evaluationResult.responseAssessmentsByName)}
              onUpsertAssessment={onUpsertAssessment}
              allowEditing={toBeReviewed}
              allowMoreThanOne
              options={KnownEvaluationResponseAssessmentNames}
              assessmentInfos={assessmentInfos}
            />
          </div>
        </div>
        <div
          css={{
            position: 'sticky',
            backgroundColor: theme.colors.backgroundSecondary,
            padding: theme.spacing.md,
            top: 0,
            display: 'flex',
            justifyContent: 'space-between',
            gap: theme.spacing.sm,
            zIndex: theme.options.zIndexBase,
          }}
        >
          <div>
            {!toBeReviewed && !isReadOnly && (
              <div
                css={{
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.general.borderRadiusBase,
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                }}
              >
                <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
                <Typography.Hint>
                  <FormattedMessage
                    defaultMessage="Reviewed"
                    description="Evaluation review > assessments > already reviewed indicator"
                  />
                </Typography.Hint>
                <Tooltip
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluations_components_evaluationsreviewassessmentssection.tsx_149"
                  content={reopenReviewTooltip}
                >
                  <Button
                    aria-label={reopenReviewTooltip}
                    componentId="mlflow.evaluations_review.reopen_review_button"
                    size="small"
                    icon={<PencilIcon />}
                    onClick={() => setOverridingExistingReview(true)}
                  />
                </Tooltip>
              </div>
            )}
          </div>
          <div css={{ flex: 1 }} />
          {pendingAssessments.length > 0 && (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <FormattedMessage
                defaultMessage="{pendingCount} pending {pendingCount, plural, =1 {change} other {changes}}"
                description="Evaluation review > assessments > pending entries counter"
                values={{ pendingCount: pendingAssessments.length }}
              />
              <Button
                componentId="mlflow.evaluations_review.discard_pending_assessments_button"
                onClick={onResetPendingAssessments}
              >
                <FormattedMessage
                  defaultMessage="Discard"
                  description="Evaluation review > assessments > discard pending assessments button label"
                />
              </Button>
            </div>
          )}
          <EvaluationsReviewAssessmentsConfirmButton
            onSave={async () => {
              // Save the pending assessments
              await onSavePendingAssessments();
              // We can reset the override review flag now
              setOverridingExistingReview(false);
            }}
            containsOverallAssessment={Boolean(currentOverallAssessment)}
            isNextResultAvailable={Boolean(isNextAvailable)}
            onClickNext={onClickNext}
            toBeReviewed={toBeReviewed}
            hasPendingAssessments={pendingAssessments.length > 0}
            overridingExistingReview={overridingExistingReview}
            onCancelOverride={() => setOverridingExistingReview(false)}
          />
        </div>
      </div>
      <Spacer size="md" />
    </div>
  );
};

/**
 * Displays section with a list of evaluation assessments: overall and detailed.
 */
export const EvaluationsReviewAssessmentsSection = ({
  evaluationResult,
  otherEvaluationResult,
  onUpsertAssessment,
  onSavePendingAssessments,
  onClickNext,
  onResetPendingAssessments,
  isNextAvailable,
  overridingExistingReview = false,
  pendingAssessments = [],
  setOverridingExistingReview,
  isReadOnly = false,
  assessmentInfos,
}: {
  evaluationResult?: RunEvaluationTracesDataEntry;
  otherEvaluationResult?: RunEvaluationTracesDataEntry;
  onUpsertAssessment: (assessment: RunEvaluationResultAssessmentDraft) => void;
  onSavePendingAssessments: () => Promise<void>;
  onClickNext?: () => void;
  onResetPendingAssessments?: () => void;
  isNextAvailable?: boolean;
  overridingExistingReview?: boolean;
  pendingAssessments?: RunEvaluationResultAssessmentDraft[];
  setOverridingExistingReview: (override: boolean) => void;
  isReadOnly?: boolean;
  assessmentInfos: AssessmentInfo[];
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        width: '100%',
        gap: theme.spacing.sm,
      }}
    >
      {evaluationResult && (
        <EvaluationsReviewSingleRunAssessmentsSection
          evaluationResult={evaluationResult}
          onUpsertAssessment={onUpsertAssessment}
          onSavePendingAssessments={onSavePendingAssessments}
          onClickNext={onClickNext}
          onResetPendingAssessments={onResetPendingAssessments}
          isNextAvailable={isNextAvailable}
          overridingExistingReview={overridingExistingReview}
          setOverridingExistingReview={setOverridingExistingReview}
          pendingAssessments={pendingAssessments}
          isReadOnly={isReadOnly}
          assessmentInfos={assessmentInfos}
        />
      )}
      {otherEvaluationResult && (
        <>
          <VerticalBar />
          <EvaluationsReviewSingleRunAssessmentsSection
            evaluationResult={otherEvaluationResult}
            onUpsertAssessment={onUpsertAssessment}
            onSavePendingAssessments={onSavePendingAssessments}
            onClickNext={onClickNext}
            onResetPendingAssessments={onResetPendingAssessments}
            isNextAvailable={isNextAvailable}
            overridingExistingReview={overridingExistingReview}
            setOverridingExistingReview={setOverridingExistingReview}
            pendingAssessments={pendingAssessments}
            isReadOnly
            assessmentInfos={assessmentInfos}
          />
        </>
      )}
    </div>
  );
};
