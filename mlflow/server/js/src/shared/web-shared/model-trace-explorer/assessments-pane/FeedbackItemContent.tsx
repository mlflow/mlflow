import { isNil } from 'lodash';
import { useState } from 'react';

import { Typography, useDesignSystemTheme, NewWindowIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';

import { AssessmentDisplayValue } from './AssessmentDisplayValue';
import { FeedbackErrorItem } from './FeedbackErrorItem';
import { FeedbackHistoryModal } from './FeedbackHistoryModal';
import { SpanNameDetailViewLink } from './SpanNameDetailViewLink';
import type { FeedbackAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { Link, useParams } from '../RoutingUtils';
import { MLFLOW_ASSESSMENT_JUDGE_COST, MLFLOW_ASSESSMENT_SCORER_TRACE_ID } from '../constants';
import { getExperimentPageTracesTabRoute } from '../routes';

export const FeedbackItemContent = ({ feedback }: { feedback: FeedbackAssessment }) => {
  const [isHistoryModalVisible, setIsHistoryModalVisible] = useState(false);
  const { theme } = useDesignSystemTheme();
  const { nodeMap, activeView } = useModelTraceExplorerViewState();
  const { experimentId } = useParams();

  const value = feedback.feedback.value;

  const associatedSpan = feedback.span_id ? nodeMap[feedback.span_id] : null;
  // the summary view displays all assessments regardless of span, so
  // we need some way to indicate which span an assessment is associated with.
  const showAssociatedSpan = activeView === 'summary' && associatedSpan;

  const judgeTraceId = feedback.metadata?.[MLFLOW_ASSESSMENT_SCORER_TRACE_ID];
  const judgeTraceHref = judgeTraceId && experimentId ? getJudgeTraceHref(experimentId, judgeTraceId) : undefined;

  const judgeCost = feedback.metadata?.[MLFLOW_ASSESSMENT_JUDGE_COST];
  const formattedCost = (() => {
    if (judgeCost === null) {
      return undefined;
    }

    const numericCost = Number(judgeCost);
    if (!Number.isFinite(numericCost)) {
      return undefined;
    }

    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(numericCost);
  })();
  const shouldShowCostSection = Boolean(formattedCost);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginLeft: theme.spacing.lg }}>
      {!isNil(feedback.feedback.error) && <FeedbackErrorItem error={feedback.feedback.error} />}
      {showAssociatedSpan && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage defaultMessage="Span" description="Label for the associated span of an assessment" />
          </Typography.Text>
          <SpanNameDetailViewLink node={associatedSpan} />
        </div>
      )}
      {isNil(feedback.feedback.error) && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage defaultMessage="Feedback" description="Label for the value of an feedback assessment" />
          </Typography.Text>
          <div css={{ display: 'flex', gap: theme.spacing.xs }}>
            <AssessmentDisplayValue jsonValue={JSON.stringify(value)} />
            {feedback.overriddenAssessment && (
              <>
                <span onClick={() => setIsHistoryModalVisible(true)}>
                  <Typography.Text
                    css={{
                      '&:hover': {
                        textDecoration: 'underline',
                        cursor: 'pointer',
                      },
                    }}
                    color="secondary"
                  >
                    <FormattedMessage
                      defaultMessage="(edited)"
                      description="Link text in an edited assessment that allows the user to click to see the previous value"
                    />
                  </Typography.Text>
                </span>
                <FeedbackHistoryModal
                  isModalVisible={isHistoryModalVisible}
                  setIsModalVisible={setIsHistoryModalVisible}
                  feedback={feedback}
                />
              </>
            )}
          </div>
        </div>
      )}
      {feedback.rationale && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="Rationale"
              description="Label for the rationale of an expectation assessment"
            />
          </Typography.Text>
          <div css={{ '& > div:last-of-type': { marginBottom: 0 } }}>
            <GenAIMarkdownRenderer>{feedback.rationale}</GenAIMarkdownRenderer>
          </div>
        </div>
      )}
      {shouldShowCostSection && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="Cost"
              description="Label for the cost metadata associated with a judge feedback"
            />
          </Typography.Text>
          <Typography.Text style={{ color: theme.colors.textSecondary }}>{formattedCost}</Typography.Text>
        </div>
      )}
      {judgeTraceHref && (
        <Link to={judgeTraceHref} target="_blank" rel="noreferrer">
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="View trace"
              description="Link text for navigating to the corresponding judge trace"
            />
            <NewWindowIcon css={{ fontSize: 12 }} />
          </span>
        </Link>
      )}
    </div>
  );
};

const getJudgeTraceHref = (experimentId: string, judgeTraceId: string) => {
  return `${getExperimentPageTracesTabRoute(experimentId)}?selectedEvaluationId=${judgeTraceId}`;
};
