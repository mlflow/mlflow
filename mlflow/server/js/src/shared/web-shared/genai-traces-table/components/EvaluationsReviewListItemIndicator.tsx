import { QuestionMarkIcon, SparkleDoubleIcon, UserIcon, useDesignSystemTheme } from '@databricks/design-system';

import { hasBeenEditedByHuman } from './GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, RunEvaluationResultAssessment } from '../types';
import { getEvaluationResultAssessmentBackgroundColor } from '../utils/Colors';

/**
 * A small indicator that shows the evaluation result's icon and sentiment.
 */
export const EvaluationsReviewListItemIndicator = ({
  assessment,
  chunkRelevanceAssessmentInfo,
}: {
  assessment?: RunEvaluationResultAssessment;
  chunkRelevanceAssessmentInfo?: AssessmentInfo;
}) => {
  const { theme } = useDesignSystemTheme();

  if (!assessment && !chunkRelevanceAssessmentInfo) {
    return <></>;
  }

  return (
    <div
      css={{
        paddingLeft: theme.spacing.sm,
        paddingRight: theme.spacing.sm,
        paddingTop: 1,
        paddingBottom: 1,
        backgroundColor: chunkRelevanceAssessmentInfo
          ? getEvaluationResultAssessmentBackgroundColor(theme, chunkRelevanceAssessmentInfo, assessment)
          : '',
        borderRadius: theme.general.borderRadiusBase,
        svg: { width: 12, height: 12 },
      }}
    >
      {assessment ? (
        <>{hasBeenEditedByHuman(assessment) ? <UserIcon /> : <SparkleDoubleIcon />}</>
      ) : (
        <QuestionMarkIcon />
      )}
    </div>
  );
};
