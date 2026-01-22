import { isEmpty, isNil, partition, some, uniqBy } from 'lodash';
import { useCallback, useMemo, useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  CloseIcon,
  PlayIcon,
  PlusIcon,
  Spacer,
  Spinner,
  TableSkeleton,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { AssessmentCreateButton } from './AssessmentCreateButton';
import { ASSESSMENT_PANE_MIN_WIDTH } from './AssessmentsPane.utils';
import { ExpectationItem } from './ExpectationItem';
import { FeedbackGroup } from './FeedbackGroup';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment, FeedbackAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useTraceCachedActions } from '../hooks/useTraceCachedActions';
import { AssessmentCreateForm } from './AssessmentCreateForm';
import { FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
import { useQueryClient } from '@tanstack/react-query';
import { invalidateMlflowSearchTracesCache } from '../hooks/invalidateMlflowSearchTracesCache';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';
import { AssessmentEditForm } from './AssessmentEditForm';

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

  return Object.entries(aggregated).toSorted(([leftName, leftFeedback], [rightName, rightFeedback]) => {
    const leftIsLLMJudge = some(leftFeedback, (feedbacks) =>
      feedbacks.some((feedback) => feedback.source.source_type === 'LLM_JUDGE'),
    );
    const rightIsLLMJudge = some(rightFeedback, (feedbacks) =>
      feedbacks.some((feedback) => feedback.source.source_type === 'LLM_JUDGE'),
    );
    if (leftIsLLMJudge && !rightIsLLMJudge) {
      return -1;
    }
    if (!leftIsLLMJudge && rightIsLLMJudge) {
      return 1;
    }
    return leftName.localeCompare(rightName);
  });
};

export const AssessmentsPane = ({
  assessments,
  traceId,
  activeSpanId,
  className,
  assessmentsTitleOverride,
  disableCloseButton,
}: {
  assessments: Assessment[];
  traceId: string;
  activeSpanId?: string;
  className?: string;
  assessmentsTitleOverride?: (count?: number) => JSX.Element;
  disableCloseButton?: boolean;
}) => {
  const reconstructAssessments = useTraceCachedActions((state) => state.reconstructAssessments);
  const cachedActions = useTraceCachedActions((state) => state.assessmentActions[traceId]);
  const { runJudgeContext, invalidateTraceQuery } = useModelTraceExplorerUpdateTraceContext();

  const queryClient = useQueryClient();
  const onRunJudgeFinishedCallback = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: [FETCH_TRACE_INFO_QUERY_KEY, traceId] });
    invalidateTraceQuery?.(traceId);
    invalidateMlflowSearchTracesCache({ queryClient });
  }, [traceId, invalidateTraceQuery, queryClient]);

  // Combine the initial assessments with the cached actions (additions and deletions)
  const allAssessments = useMemo(() => {
    // Caching actions is enabled only with Traces V4 feature
    if (!shouldUseTracesV4API()) {
      return assessments;
    }
    const reconstructed = reconstructAssessments(assessments, cachedActions);
    return uniqBy(reconstructed, ({ assessment_id }) => assessment_id);
  }, [assessments, reconstructAssessments, cachedActions]);

  const { theme } = useDesignSystemTheme();
  const { setAssessmentsPaneExpanded, assessmentsPaneExpanded, isInComparisonView } = useModelTraceExplorerViewState();
  const [feedbacks, expectations] = useMemo(
    () => partition(allAssessments, (assessment) => 'feedback' in assessment),
    [allAssessments],
  );
  const groupedFeedbacks = useMemo(() => {
    return groupFeedbacks(feedbacks);
  }, [feedbacks]);
  const sortedExpectations = expectations.toSorted((left, right) =>
    left.assessment_name.localeCompare(right.assessment_name),
  );

  const [createFormDisplayMode, setCreateFormDisplayMode] = useState<null | 'feedback' | 'expectation'>(null);

  return (
    <div
      data-testid="assessments-pane"
      css={{
        display: 'flex',
        flexDirection: 'column',
        ...(isInComparisonView
          ? { padding: `${theme.spacing.sm} 0`, maxHeight: theme.spacing.lg * 10 }
          : { padding: theme.spacing.sm, paddingTop: theme.spacing.xs, height: '100%' }),
        ...(isInComparisonView ? {} : { borderLeft: `1px solid ${theme.colors.border}` }),
        overflowY: 'auto',
        minWidth: ASSESSMENT_PANE_MIN_WIDTH,
        width: '100%',
        boxSizing: 'border-box',
      }}
      className={className}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.sm,
        }}
      >
        {!isInComparisonView &&
          (assessmentsTitleOverride ? (
            assessmentsTitleOverride()
          ) : (
            <FormattedMessage defaultMessage="Assessments" description="Label for the assessments pane" />
          ))}
        {!isInComparisonView && setAssessmentsPaneExpanded && !disableCloseButton && (
          <Tooltip
            componentId="shared.model-trace-explorer.close-assessments-pane-tooltip"
            content={
              <FormattedMessage
                defaultMessage="Hide assessments"
                description="Tooltip for a button that closes the assessments pane"
              />
            }
          >
            <Button
              data-testid="close-assessments-pane-button"
              componentId="shared.model-trace-explorer.close-assessments-pane"
              size="small"
              icon={<CloseIcon />}
              onClick={() => setAssessmentsPaneExpanded(false)}
            />
          </Tooltip>
        )}
      </div>
      {/* <AssessmentJudgesSection traceId={traceId} groupedFeedbacks={groupedJudgeFeedbacks} /> */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.sm,
          height: theme.spacing.lg,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Feedback"
            description="Label for the feedback section in the assessments pane"
          />{' '}
          {!isEmpty(groupedFeedbacks) && <>({groupedFeedbacks?.length})</>}
        </Typography.Text>
      </div>
      {!isEmpty(groupedFeedbacks) && (
        <div css={{ display: 'flex', gap: 4, justifyContent: 'flex-end', marginBottom: theme.spacing.sm }}>
          <Button
            componentId="shared.model-trace-explorer.add-feedback"
            size="small"
            icon={<PlusIcon />}
            onClick={() => setCreateFormDisplayMode('feedback')}
          >
            <FormattedMessage
              defaultMessage="Add feedback"
              description="Label for the button to add a new assessment"
            />
          </Button>
          {runJudgeContext?.renderRunJudgeButton({
            traceId,
            onRunJudgeFinishedCallback,
            disabled: runJudgeContext?.judgeExecutionState.isLoading,
          })}
        </div>
      )}
      {runJudgeContext?.judgeExecutionState.isLoading && (
        <div
          css={{
            borderRadius: theme.spacing.sm,
            border: `1px solid ${theme.colors.border}`,
            padding: 12,
            paddingTop: 8,
            marginBottom: theme.spacing.sm,
          }}
        >
          <Typography.Text bold>{runJudgeContext?.judgeExecutionState.scorerInProgress}</Typography.Text>
          <TableSkeleton lines={4} />
        </div>
      )}
      {groupedFeedbacks.map(([name, valuesMap]) => (
        <FeedbackGroup key={name} name={name} valuesMap={valuesMap} traceId={traceId} activeSpanId={activeSpanId} />
      ))}
      {isEmpty(groupedFeedbacks) && (
        <div
          css={{
            textAlign: 'center',
            borderRadius: theme.spacing.sm,
            border: `1px solid ${theme.colors.border}`,
            padding: `${theme.spacing.md}px ${theme.spacing.md}px`,
          }}
        >
          <Typography.Hint>
            Add a custom feedback to this trace. <Typography.Link componentId="TODO">Learn more.</Typography.Link>
          </Typography.Hint>
          <Spacer size="sm" />
          <div css={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
            <Button
              componentId="TODO"
              size="small"
              icon={<PlusIcon />}
              onClick={() => setCreateFormDisplayMode('feedback')}
            >
              Add feedback
            </Button>
            {runJudgeContext?.renderRunJudgeButton({
              traceId,
              onRunJudgeFinishedCallback,
              disabled: runJudgeContext?.judgeExecutionState.isLoading,
            })}
          </div>
        </div>
      )}
      {createFormDisplayMode === 'feedback' && (
        <AssessmentCreateForm
          spanId={activeSpanId}
          traceId={traceId}
          initialAssessmentType="feedback"
          setExpanded={() => setCreateFormDisplayMode(null)}
        />
      )}
      <Spacer size="sm" />
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.sm,
          height: theme.spacing.lg,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Expectations"
            description="Label for the expectations section in the assessments pane"
          />{' '}
          {!isEmpty(sortedExpectations) && <>({sortedExpectations?.length})</>}
        </Typography.Text>
        {!isEmpty(sortedExpectations) && (
          <Button
            componentId="shared.model-trace-explorer.add-expectation"
            size="small"
            icon={<PlusIcon />}
            onClick={() => setCreateFormDisplayMode('expectation')}
          >
            <FormattedMessage
              defaultMessage="Add expectation"
              description="Label for the button to add a new expectation"
            />
          </Button>
        )}
      </div>
      {sortedExpectations.length > 0 ? (
        <>
          <div
            css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}
          >
            {sortedExpectations.map((expectation) => (
              <ExpectationItem expectation={expectation} key={expectation.assessment_id} />
            ))}
          </div>
        </>
      ) : (
        <div
          css={{
            textAlign: 'center',
            borderRadius: theme.spacing.sm,
            border: `1px solid ${theme.colors.border}`,
            padding: `${theme.spacing.md}px ${theme.spacing.md}px`,
          }}
        >
          <Typography.Hint>
            Add a custom expectation to this trace. <Typography.Link componentId="TODO">Learn more.</Typography.Link>
          </Typography.Hint>
          <Spacer size="sm" />
          <Button
            componentId="TODO"
            size="small"
            icon={<PlusIcon />}
            onClick={() => setCreateFormDisplayMode('expectation')}
          >
            Add expectation
          </Button>
        </div>
      )}
      {createFormDisplayMode === 'expectation' && (
        <AssessmentCreateForm
          spanId={activeSpanId}
          traceId={traceId}
          initialAssessmentType="expectation"
          setExpanded={() => setCreateFormDisplayMode(null)}
        />
      )}
    </div>
  );
};
