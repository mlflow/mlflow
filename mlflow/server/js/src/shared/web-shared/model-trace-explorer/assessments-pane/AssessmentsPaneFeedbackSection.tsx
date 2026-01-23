import { Button, PlusIcon, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FeedbackAssessment } from '../ModelTrace.types';
import { FeedbackGroup } from './FeedbackGroup';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { first, isEmpty, isNil, partition, some } from 'lodash';
import { AssessmentCreateForm } from './AssessmentCreateForm';
import { AssessmentsPaneJudgeFeedbackSection } from './AssessmentsPaneJudgeFeedbackSection';
import { AssessmentSourceTypeTag } from './AssessmentSourceTypeTag';
import { AssessmentSourceTypeTagList } from './AssessmentSourceTypeTagList';

type GroupedFeedbacksByValue = { [value: string]: FeedbackAssessment[] };

type GroupedFeedbacks = [assessmentName: string, feedbacks: GroupedFeedbacksByValue][];

// Helper function to check if any feedback in a group is from an LLM judge or code scorer
const hasAutomatedFeedback = (feedbacksMap: GroupedFeedbacksByValue): boolean =>
  Object.values(feedbacksMap)
    .flat()
    .some((feedback) => feedback.source.source_type === 'LLM_JUDGE' || feedback.source.source_type === 'CODE');

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
  // TODO(next PR): Extract LLM judge feedback groups to a separate section
  return Object.entries(aggregated);
};

const AddFeedbackButton = ({ onClick }: { onClick: () => void }) => (
  <Button
    type="primary"
    componentId="shared.model-trace-explorer.add-feedback"
    size="small"
    icon={<PlusIcon />}
    onClick={onClick}
  >
    <FormattedMessage defaultMessage="Add human feedback" description="Label for the button to add a new feedback" />
  </Button>
);

export const AssessmentsPaneFeedbackSection = ({
  showRunScorerSection,
  feedbacks,
  activeSpanId,
  traceId,
}: {
  showRunScorerSection: boolean;
  feedbacks: FeedbackAssessment[];
  activeSpanId?: string;
  traceId: string;
}) => {
  const [groupedFeedbacks, llmJudgeFeedbacks] = useMemo(() => {
    const groups = groupFeedbacks(feedbacks);
    if (!showRunScorerSection) {
      return [groups, []];
    }
    // Break out the groups into two lists: one with groups that have automated feedback and one without
    return partition(groups, ([, feedbacks]) => !hasAutomatedFeedback(feedbacks));
  }, [feedbacks, showRunScorerSection]);

  const [createFormVisible, setCreateFormVisible] = useState(false);

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
      {showRunScorerSection && (
        <>
          <AssessmentsPaneJudgeFeedbackSection
            feedbackGroups={llmJudgeFeedbacks}
            traceId={traceId}
            activeSpanId={activeSpanId}
          />
          <Spacer size="sm" shrinks={false} />
        </>
      )}
      {!isEmpty(groupedFeedbacks) && (
        <div css={{ display: 'flex', justifyContent: 'flex-end', marginBottom: theme.spacing.sm }}>
          <AddFeedbackButton onClick={() => setCreateFormVisible(true)} />
        </div>
      )}

      {groupedFeedbacks.map(([name, valuesMap]) => (
        <FeedbackGroup
          key={name}
          name={name}
          valuesMap={valuesMap}
          traceId={traceId}
          activeSpanId={activeSpanId}
          feedbackTypeTag={<AssessmentSourceTypeTagList assessments={Object.values(valuesMap).flat()} />}
        />
      ))}
      {isEmpty(groupedFeedbacks) && !createFormVisible && (
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
            <AddFeedbackButton onClick={() => setCreateFormVisible(true)} />
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
