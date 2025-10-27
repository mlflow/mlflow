import { isNil, partition } from 'lodash';
import { useMemo } from 'react';

import { Button, CloseIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { AssessmentCreateButton } from './AssessmentCreateButton';
import { ASSESSMENT_PANE_MIN_WIDTH } from './AssessmentsPane.utils';
import { ExpectationItem } from './ExpectationItem';
import { FeedbackGroup } from './FeedbackGroup';
import type { Assessment, FeedbackAssessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

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

  return Object.entries(aggregated).toSorted(([leftName], [rightName]) => leftName.localeCompare(rightName));
};

export const AssessmentsPane = ({
  assessments,
  traceId,
  activeSpanId,
}: {
  assessments: Assessment[];
  traceId: string;
  activeSpanId?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { setAssessmentsPaneExpanded } = useModelTraceExplorerViewState();
  const [feedbacks, expectations] = useMemo(
    () => partition(assessments, (assessment) => 'feedback' in assessment),
    [assessments],
  );
  const groupedFeedbacks = useMemo(() => groupFeedbacks(feedbacks), [feedbacks]);
  const sortedExpectations = expectations.toSorted((left, right) =>
    left.assessment_name.localeCompare(right.assessment_name),
  );

  return (
    <div
      data-testid="assessments-pane"
      css={{
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.sm,
        paddingTop: theme.spacing.xs,
        height: '100%',
        borderLeft: `1px solid ${theme.colors.border}`,
        overflowY: 'scroll',
        minWidth: ASSESSMENT_PANE_MIN_WIDTH,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between' }}>
        <Typography.Text css={{ marginBottom: theme.spacing.sm }} bold>
          <FormattedMessage defaultMessage="Assessments" description="Label for the assessments pane" />
        </Typography.Text>
        {setAssessmentsPaneExpanded && (
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
      {groupedFeedbacks.map(([name, valuesMap]) => (
        <FeedbackGroup key={name} name={name} valuesMap={valuesMap} traceId={traceId} activeSpanId={activeSpanId} />
      ))}
      {sortedExpectations.length > 0 && (
        <>
          <Typography.Text color="secondary" css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Expectations"
              description="Label for the expectations section in the assessments pane"
            />
          </Typography.Text>
          <div
            css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}
          >
            {sortedExpectations.map((expectation) => (
              <ExpectationItem expectation={expectation} key={expectation.assessment_id} />
            ))}
          </div>
        </>
      )}
      <AssessmentCreateButton
        title={
          <FormattedMessage
            defaultMessage="Add new assessment"
            description="Label for the button to add a new assessment"
          />
        }
        spanId={activeSpanId}
        traceId={traceId}
      />
    </div>
  );
};
