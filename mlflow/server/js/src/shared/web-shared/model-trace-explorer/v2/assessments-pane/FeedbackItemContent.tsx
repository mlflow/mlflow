import { isNil } from 'lodash';
import { useMemo, useState } from 'react';

import { FileDocumentIcon, Typography, useDesignSystemTheme, NewWindowIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '../../../genai-markdown-renderer/GenAIMarkdownRenderer';

import { AssessmentDisplayValue } from '../../assessments-pane/AssessmentDisplayValue';
import { FeedbackErrorItem } from '../../assessments-pane/FeedbackErrorItem';
import { FeedbackHistoryModal } from '../../assessments-pane/FeedbackHistoryModal';
import type { FeedbackAssessment, RetrieverDocument } from '../ModelTrace.types';
import { getAssessmentDocumentIndex, isChunkRelevanceAssessment } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { Link, useParams } from '../../RoutingUtils';
import { MLFLOW_ASSESSMENT_JUDGE_COST, MLFLOW_ASSESSMENT_SCORER_TRACE_ID } from '../../constants';
import { getExperimentPageTracesTabRoute } from '../../routes';
import { formatCostUSD } from '../../CostUtils';

export const FeedbackItemContent = ({ feedback }: { feedback: FeedbackAssessment }): JSX.Element => {
  const [isHistoryModalVisible, setIsHistoryModalVisible] = useState(false);
  const { theme } = useDesignSystemTheme();
  const { nodeMap } = useModelTraceExplorerViewState();
  const { experimentId } = useParams();

  const value = feedback.feedback.value;

  const associatedSpan = feedback.span_id ? nodeMap[feedback.span_id] : null;
  const documentPreview = useMemo(() => {
    if (!isChunkRelevanceAssessment(feedback) || !associatedSpan?.outputs) return null;
    const documentIndex = getAssessmentDocumentIndex(feedback);
    if (documentIndex === undefined) return null;
    const outputs = associatedSpan.outputs as RetrieverDocument[];
    if (!Array.isArray(outputs) || documentIndex >= outputs.length) return null;
    return outputs[documentIndex]?.page_content ?? null;
  }, [feedback, associatedSpan]);

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

    return formatCostUSD(numericCost);
  })();
  const shouldShowCostSection = Boolean(formattedCost);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginLeft: theme.spacing.lg }}>
      {!isNil(feedback.feedback.error) && <FeedbackErrorItem error={feedback.feedback.error} />}
      {documentPreview && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="Doc"
              description="Label for the document preview in a chunk relevance assessment"
            />
          </Typography.Text>
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              padding: theme.spacing.xs,
              borderRadius: theme.borders.borderRadiusSm,
              backgroundColor: theme.colors.backgroundSecondary,
            }}
          >
            <FileDocumentIcon css={{ flexShrink: 0, color: theme.colors.textSecondary }} />
            <Typography.Text ellipsis size="sm" css={{ color: theme.colors.textSecondary }}>
              {documentPreview}
            </Typography.Text>
          </div>
        </div>
      )}
      {isNil(feedback.feedback.error) && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage defaultMessage="Feedback" description="Label for the value of an feedback assessment" />
          </Typography.Text>
          <div css={{ display: 'flex', gap: theme.spacing.xs }}>
            <AssessmentDisplayValue jsonValue={JSON.stringify(value)} assessmentName={feedback.assessment_name} />
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
            <GenAIMarkdownRenderer compact>{feedback.rationale}</GenAIMarkdownRenderer>
          </div>
        </div>
      )}
      {shouldShowCostSection && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="Cost"
              description="Label for the cost metadata associated with scorer feedback"
            />
          </Typography.Text>
          <Typography.Text style={{ color: theme.colors.textSecondary }}>{formattedCost}</Typography.Text>
        </div>
      )}
      {judgeTraceHref && (
        <Link
          componentId="mlflow.model_trace_explorer.feedback_item.judge_trace_link"
          to={judgeTraceHref}
          target="_blank"
          rel="noreferrer"
        >
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="View trace"
              description="Link text for navigating to the corresponding scorer trace"
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
