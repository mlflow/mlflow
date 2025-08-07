import { Button, ChevronRightIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export const EvaluationsReviewAssessmentsConfirmButton = ({
  toBeReviewed,
  containsOverallAssessment,
  isNextResultAvailable,
  overridingExistingReview,
  hasPendingAssessments,
  onClickNext,
  onSave,
  onCancelOverride,
}: {
  toBeReviewed: boolean;
  containsOverallAssessment: boolean;
  isNextResultAvailable: boolean;
  hasPendingAssessments: boolean;
  overridingExistingReview: boolean;
  onSave?: () => Promise<void>;
  onClickNext?: () => void;
  onCancelOverride?: () => void;
}) => {
  if (toBeReviewed) {
    if (hasPendingAssessments) {
      return (
        <Button type="primary" componentId="mlflow.evaluations_review.save_pending_assessments_button" onClick={onSave}>
          <FormattedMessage defaultMessage="Save" description="Evaluation review > assessments > save button" />
        </Button>
      );
    }
    if (overridingExistingReview) {
      return (
        <Button componentId="mlflow.evaluations_review.cancel_override_assessments_button" onClick={onCancelOverride}>
          <FormattedMessage
            defaultMessage="Cancel"
            description="Evaluation review > assessments > cancel overriding review button"
          />
        </Button>
      );
    }
    return (
      <Button
        type="primary"
        componentId="mlflow.evaluations_review.mark_as_reviewed_button"
        onClick={onSave}
        disabled={!containsOverallAssessment}
      >
        <FormattedMessage
          defaultMessage="Mark as reviewed"
          description="Evaluation review > assessments > mark as reviewed button"
        />
      </Button>
    );
  }
  return (
    <Button
      type="primary"
      componentId="mlflow.evaluations_review.next_evaluation_result_button"
      onClick={onClickNext}
      endIcon={<ChevronRightIcon />}
      disabled={!isNextResultAvailable}
    >
      <FormattedMessage defaultMessage="Next" description="Evaluation review > assessments > next button" />
    </Button>
  );
};
