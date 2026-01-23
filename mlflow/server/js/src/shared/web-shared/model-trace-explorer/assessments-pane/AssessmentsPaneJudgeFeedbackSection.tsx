import { Button, PlusIcon, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FeedbackAssessment } from '../ModelTrace.types';
import { FeedbackGroup } from './FeedbackGroup';
import { FormattedMessage } from 'react-intl';
import { first, isEmpty } from 'lodash';
import { AssessmentSourceTypeTag } from './AssessmentSourceTypeTag';
import { AssessmentSourceTypeTagList } from './AssessmentSourceTypeTagList';

type GroupedFeedbacksByValue = { [value: string]: FeedbackAssessment[] };

type GroupedFeedbacks = [assessmentName: string, feedbacks: GroupedFeedbacksByValue][];

const RunScorerButton = () => (
  <Button type="primary" componentId="shared.model-trace-explorer.add-feedback" size="small" icon={<PlusIcon />}>
    {/* TODO (next PR): Implement selecting and running a judge */}
    <FormattedMessage defaultMessage="Run judge" description="Label for the button to add a new feedback" />
  </Button>
);

export const AssessmentsPaneJudgeFeedbackSection = ({
  feedbackGroups,
  activeSpanId,
  traceId,
}: {
  feedbackGroups: GroupedFeedbacks;
  activeSpanId?: string;
  traceId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <>
      {!isEmpty(feedbackGroups) && (
        <div css={{ display: 'flex', justifyContent: 'flex-end', marginBottom: theme.spacing.sm }}>
          <RunScorerButton />
        </div>
      )}
      {isEmpty(feedbackGroups) && (
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
              defaultMessage="Run a scorer to get feedback on this trace."
              description="Hint message prompting user to run a scorer"
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
          <div css={{ display: 'flex', justifyContent: 'center' }}>
            <RunScorerButton />
          </div>
        </div>
      )}
      {feedbackGroups.map(([name, valuesMap]) => (
        <FeedbackGroup
          key={name}
          name={name}
          valuesMap={valuesMap}
          traceId={traceId}
          activeSpanId={activeSpanId}
          feedbackTypeTag={<AssessmentSourceTypeTagList assessments={Object.values(valuesMap).flat()} />}
        />
      ))}
    </>
  );
};
