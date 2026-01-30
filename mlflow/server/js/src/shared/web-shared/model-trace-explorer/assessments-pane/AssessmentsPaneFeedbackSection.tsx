import {
  Alert,
  Button,
  DropdownMenu,
  GavelIcon,
  PlusIcon,
  Spacer,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FeedbackAssessment } from '../ModelTrace.types';
import { FeedbackGroup } from './FeedbackGroup';
import { useEffect, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { isEmpty, isNil, uniqBy } from 'lodash';
import { AssessmentCreateForm } from './AssessmentCreateForm';
import { useModelTraceExplorerRunJudgesContext } from '../contexts/RunJudgesContext';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';
import { useQueryClient } from '@tanstack/react-query';
import { invalidateMlflowSearchTracesCache } from '../hooks/invalidateMlflowSearchTracesCache';
import { FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
import { isEvaluatingTracesInDetailsViewEnabled } from '../FeatureUtils';

type GroupedFeedbacksByValue = { [value: string]: FeedbackAssessment[] };

type GroupedFeedbacks = [assessmentName: string, feedbacks: GroupedFeedbacksByValue][];

const groupFeedbacks = (feedbacks: FeedbackAssessment[]): GroupedFeedbacks => {
  const aggregated: Record<string, GroupedFeedbacksByValue> = {};
  feedbacks.forEach((feedback) => {
    if (feedback.valid === false) {
      return;
    }

    let value = null;
    if (feedback.feedback.value !== '') {
      value = JSON.stringify(feedback.feedback.value);
    }

    const { assessment_name } = feedback;
    if (!aggregated[assessment_name]) {
      aggregated[assessment_name] = {};
    }

    const group = aggregated[assessment_name];
    if (!isNil(value)) {
      if (!group[value]) {
        group[value] = [];
      }
      group[value].push(feedback);
    }
  });

  // Filter out LLM judge feedback groups
  return Object.entries(aggregated);
};

const AddFeedbackButton = ({
  onClick,
  traceId,
  sessionId,
}: {
  onClick: () => void;
  traceId: string;
  sessionId?: string;
}) => {
  const runJudgeConfiguration = useModelTraceExplorerRunJudgesContext();
  const [judgeModalVisible, setJudgeModalVisible] = useState(false);

  if (runJudgeConfiguration.renderRunJudgeModal && isEvaluatingTracesInDetailsViewEnabled()) {
    return (
      <>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <Button
              type="primary"
              componentId="shared.model-trace-explorer.add-feedback"
              size="small"
              icon={<PlusIcon />}
            >
              <FormattedMessage
                defaultMessage="Add feedback"
                description="Label for the button to add a new feedback"
              />
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content>
            <DropdownMenu.Item componentId="mlflow.model-trace-explorer.add-human-feedback" onClick={onClick}>
              <FormattedMessage
                defaultMessage="Human feedback"
                description="Label for the button to add a human feedback to the trace"
              />
            </DropdownMenu.Item>
            <DropdownMenu.Item
              componentId="mlflow.model-trace-explorer.run-judge"
              onClick={() => setJudgeModalVisible(true)}
            >
              <FormattedMessage
                defaultMessage="LLM judge feedback"
                description="Label for the button to add a LLM judge feedback to the trace"
              />
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
        {runJudgeConfiguration.renderRunJudgeModal?.({
          itemId: sessionId ?? traceId,
          visible: judgeModalVisible,
          onClose: () => setJudgeModalVisible(false),
        })}
      </>
    );
  }

  return (
    <Button
      type="primary"
      componentId="shared.model-trace-explorer.add-feedback"
      size="small"
      icon={<PlusIcon />}
      onClick={onClick}
    >
      <FormattedMessage defaultMessage="Add feedback" description="Label for the button to add a new feedback" />
    </Button>
  );
};

export const AssessmentsPaneFeedbackSection = ({
  enableRunScorer,
  feedbacks,
  activeSpanId,
  traceId,
  sessionId,
}: {
  enableRunScorer: boolean;
  feedbacks: FeedbackAssessment[];
  activeSpanId?: string;
  traceId: string;
  sessionId?: string;
}) => {
  const groupedFeedbacks = useMemo(() => groupFeedbacks(feedbacks), [feedbacks]);

  const [createFormVisible, setCreateFormVisible] = useState(false);

  const { evaluations, subscribeToScorerFinished } = useModelTraceExplorerRunJudgesContext();

  // Derive evaluations for this trace from context state
  const currentTraceEvaluations = useMemo(() => {
    if (!evaluations) return [];
    return Object.values(evaluations).filter((event) => traceId in (event.tracesData ?? {}));
  }, [evaluations, traceId]);

  const { newTracePendingEvaluations, loadingEvaluations } = useMemo(() => {
    const loadingEvaluations = currentTraceEvaluations.filter((event) => event.isLoading);
    const newTracePendingEvaluations = uniqBy(
      loadingEvaluations.filter(
        (evaluation) => !groupedFeedbacks.some(([groupName]) => groupName === evaluation.label),
      ),
      'label',
    );
    return { newTracePendingEvaluations, loadingEvaluations };
  }, [currentTraceEvaluations, groupedFeedbacks]);

  const currentTraceEvaluationErrors = useMemo(
    () => currentTraceEvaluations.filter((event) => event.error),
    [currentTraceEvaluations],
  );

  const { invalidateTraceQuery } = useModelTraceExplorerUpdateTraceContext();
  const queryClient = useQueryClient();

  useEffect(() => {
    return subscribeToScorerFinished?.((event) => {
      const isCurrentTraceEvaluation = event.results?.some(
        (result) =>
          'trace' in result &&
          result.trace?.info &&
          'trace_id' in result.trace?.info &&
          result.trace?.info.trace_id === traceId,
      );
      if (!isCurrentTraceEvaluation) {
        return;
      }
      invalidateTraceQuery?.(traceId);
      queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
      invalidateMlflowSearchTracesCache({ queryClient });
    });
  }, [subscribeToScorerFinished, traceId, invalidateTraceQuery, queryClient]);

  const isSectionEmpty =
    isEmpty(groupedFeedbacks) && isEmpty(currentTraceEvaluationErrors) && isEmpty(newTracePendingEvaluations);

  const { theme } = useDesignSystemTheme();
  return (
    <>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.sm,
          height: theme.spacing.lg,
          flexShrink: 0,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Feedback"
            description="Label for the feedback section in the assessments pane"
          />{' '}
          {!isEmpty(groupedFeedbacks) && <>({feedbacks?.length})</>}
        </Typography.Text>
      </div>

      {!isSectionEmpty && (
        <div
          css={{ display: 'flex', justifyContent: 'flex-end', marginBottom: theme.spacing.sm, gap: theme.spacing.xs }}
        >
          <AddFeedbackButton traceId={traceId} sessionId={sessionId} onClick={() => setCreateFormVisible(true)} />
        </div>
      )}

      {currentTraceEvaluationErrors.map((evaluation) => (
        <div key={evaluation.requestKey} css={{ marginBottom: theme.spacing.sm }}>
          <Alert
            closable={false}
            type="error"
            message={
              <FormattedMessage
                defaultMessage='Error evaluating "{label}"'
                description="Error evaluating label"
                values={{ label: evaluation.label }}
              />
            }
            description={evaluation.error?.message}
            componentId="shared.model-trace-explorer.feedback-error-item"
          />
        </div>
      ))}
      {newTracePendingEvaluations.map((evaluation) => (
        <div
          key={evaluation.requestKey}
          css={{
            borderRadius: theme.spacing.sm,
            border: `1px solid ${theme.colors.border}`,
            padding: theme.spacing.sm + theme.spacing.xs,
            paddingTop: theme.spacing.sm,
            marginBottom: theme.spacing.sm,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {/* <AssessmentSourceTypeTag sourceType="LLM_JUDGE" /> */}
          <Typography.Text bold>{evaluation.label}</Typography.Text>
          <TableSkeleton lines={3} />
        </div>
      ))}

      {groupedFeedbacks.map(([name, valuesMap]) => (
        <FeedbackGroup
          key={name}
          name={name}
          valuesMap={valuesMap}
          traceId={traceId}
          activeSpanId={activeSpanId}
          loading={loadingEvaluations.some((evaluation) => evaluation.label === name)}
        />
      ))}
      {isSectionEmpty && !createFormVisible && (
        <div
          css={{
            textAlign: 'center',
            borderRadius: theme.spacing.xs,
            border: `1px dashed ${theme.colors.border}`,
            padding: `${theme.spacing.md}px ${theme.spacing.md}px`,
          }}
        >
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Add a custom feedback to this trace."
              description="Hint message prompting user to add a new feedback"
            />{' '}
            <Typography.Link
              componentId="shared.model-trace-explorer.feedback-learn-more-link"
              openInNewTab
              href="https://www.mlflow.org/docs/latest/genai/assessments/feedback/"
            >
              <FormattedMessage defaultMessage="Learn more." description="Link text for learning more about feedback" />
            </Typography.Link>
          </Typography.Hint>
          <Spacer size="sm" />
          <div css={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
            <AddFeedbackButton traceId={traceId} sessionId={sessionId} onClick={() => setCreateFormVisible(true)} />
          </div>
        </div>
      )}
      {createFormVisible && (
        <AssessmentCreateForm
          spanId={activeSpanId}
          traceId={traceId}
          initialAssessmentType="feedback"
          setExpanded={() => setCreateFormVisible(false)}
        />
      )}
    </>
  );
};
