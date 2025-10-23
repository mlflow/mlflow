import { isNil } from 'lodash';
import { useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';

import { AssessmentDisplayValue } from './AssessmentDisplayValue';
import { FeedbackErrorItem } from './FeedbackErrorItem';
import { FeedbackHistoryModal } from './FeedbackHistoryModal';
import { SpanNameDetailViewLink } from './SpanNameDetailViewLink';
import type { FeedbackAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const FeedbackItemContent = ({ feedback }: { feedback: FeedbackAssessment }) => {
  const [isHistoryModalVisible, setIsHistoryModalVisible] = useState(false);
  const { theme } = useDesignSystemTheme();
  const { nodeMap, activeView } = useModelTraceExplorerViewState();

  const value = feedback.feedback.value;

  const associatedSpan = feedback.span_id ? nodeMap[feedback.span_id] : null;
  // the summary view displays all assessments regardless of span, so
  // we need some way to indicate which span an assessment is associated with.
  const showAssociatedSpan = activeView === 'summary' && associatedSpan;

  const judgeTraceId = feedback.metadata?.['mlflow.assessment.scorerTraceId'];
  const judgeTraceHref = judgeTraceId ? getJudgeTraceHref(judgeTraceId) : undefined;
  const shouldShowJudgeTraceSection = Boolean(judgeTraceHref);

  const judgeCost = feedback.metadata?.['mlflow.assessment.judgeCost'];
  const formattedCost = (() => {
    if (judgeCost === null) {
      return undefined;
    }

    const numericCost = Number(judgeCost);
    if (!Number.isFinite(numericCost)) {
      return undefined;
    }

    const decimalMatch = String(judgeCost).match(/\.(\d+)/);
    const truncatedDecimals = Math.min(Math.max(decimalMatch ? decimalMatch[1].length : 0, 2), 6);

    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: truncatedDecimals,
      maximumFractionDigits: truncatedDecimals,
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
      {shouldShowJudgeTraceSection && (
        <Typography.Link
          href={judgeTraceHref}
          openInNewTab
          componentId="shared.model-trace-explorer.feedback-cost.trace-link"
        >
          <FormattedMessage
            defaultMessage="View trace"
            description="Link text for navigating to the corresponding judge trace"
          />
        </Typography.Link>
      )}
    </div>
  );
};

/**
 * Returns the href for the judge trace link.
 *
 * @param id - The ID of the judge trace.
 * @returns The href for the judge trace.
 */
const getJudgeTraceHref = (id: string) => {
  const { pathname, hash } = window.location;
  const experimentMatchFromHash = hash?.match(/\/experiments\/(\d+|[^/]+)/);
  const experimentMatchFromPath = pathname?.match(/\/experiments\/(\d+|[^/]+)/);
  const experimentId = experimentMatchFromHash?.[1] ?? experimentMatchFromPath?.[1];

  if (experimentId) {
    const basePath = `/experiments/${experimentId}/traces?selectedEvaluationId=${encodeURIComponent(id)}`;
    // If the router uses hash history, preserve it so the link works in a new tab.
    if (hash?.includes('/experiments/')) {
      return `#${basePath}`;
    }
    return basePath;
  }

  // Fallback when we cannot infer the experiment: open traces view with evaluation selection.
  if (hash) {
    return `#${`/traces?selectedEvaluationId=${encodeURIComponent(id)}`}`;
  }
  return `/traces?selectedEvaluationId=${encodeURIComponent(id)}`;
};
