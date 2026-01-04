import { isNil, partition, uniqBy } from 'lodash';
import { useMemo } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  CloseIcon,
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
  className,
  assessmentsTitleOverride,
}: {
  assessments: Assessment[];
  traceId: string;
  activeSpanId?: string;
  className?: string;
  assessmentsTitleOverride?: (count?: number) => JSX.Element;
}) => {
  const reconstructAssessments = useTraceCachedActions((state) => state.reconstructAssessments);
  const cachedActions = useTraceCachedActions((state) => state.assessmentActions[traceId]);

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
      <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        {!isInComparisonView &&
          (assessmentsTitleOverride ? (
            assessmentsTitleOverride()
          ) : (
            <FormattedMessage defaultMessage="Assessments" description="Label for the assessments pane" />
          ))}
        {!isInComparisonView && setAssessmentsPaneExpanded && (
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
