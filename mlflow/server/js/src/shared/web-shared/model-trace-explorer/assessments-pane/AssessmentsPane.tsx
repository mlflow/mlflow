import { uniqBy } from 'lodash';
import { useMemo } from 'react';

import { Button, CloseIcon, Spacer, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ASSESSMENT_PANE_MIN_WIDTH } from './AssessmentsPane.utils';
import { isEvaluatingTracesInDetailsViewEnabled, shouldUseTracesV4API } from '../FeatureUtils';
import type {
  Assessment,
  ExpectationAssessment,
  FeedbackAssessment,
  IssueReferenceAssessment,
} from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useTraceCachedActions } from '../hooks/useTraceCachedActions';
import { AssessmentsPaneExpectationsSection } from './AssessmentsPaneExpectationsSection';
import { AssessmentsPaneFeedbackSection } from './AssessmentsPaneFeedbackSection';
import { AssessmentsPaneIssuesSection } from './AssessmentsPaneIssuesSection';
import { useModelTraceExplorerRunJudgesContext } from '../contexts/RunJudgesContext';
import { useSelectedIssueId } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId';

/**
 * Safely calls useSelectedIssueId hook, returning undefined if not in a Router context.
 * This prevents crashes when AssessmentsPane is rendered outside a Router (e.g., tests, OSS notebook).
 */
const useSafeSelectedIssueId = (): string | undefined => {
  try {
    const [selectedIssueId] = useSelectedIssueId();
    return selectedIssueId;
  } catch (error) {
    // Hook throws when not in a Router context
    return undefined;
  }
};

export const AssessmentsPane = ({
  assessments,
  traceId,
  sessionId,
  activeSpanId,
  className,
  assessmentsTitleOverride,
  disableCloseButton,
  enableRunScorer = true,
}: {
  assessments: Assessment[];
  traceId: string;
  sessionId?: string;
  activeSpanId?: string;
  className?: string;
  assessmentsTitleOverride?: (count?: number) => JSX.Element;
  disableCloseButton?: boolean;
  enableRunScorer?: boolean;
}) => {
  const reconstructAssessments = useTraceCachedActions((state) => state.reconstructAssessments);
  const cachedActions = useTraceCachedActions((state) => state.assessmentActions[traceId]);

  // Get selected issue ID from URL (safe in non-router contexts)
  const selectedIssueId = useSafeSelectedIssueId();

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
  const { setAssessmentsPaneExpanded } = useModelTraceExplorerViewState();

  const { feedbacks, expectations, issues } = useMemo(() => {
    const feedbacks: FeedbackAssessment[] = [];
    const expectations: ExpectationAssessment[] = [];
    const issues: IssueReferenceAssessment[] = [];

    for (const assessment of allAssessments) {
      if ('feedback' in assessment) {
        feedbacks.push(assessment);
      } else if ('issue' in assessment) {
        issues.push(assessment);
      } else if ('expectation' in assessment) {
        expectations.push(assessment);
      }
    }

    return { feedbacks, expectations, issues };
  }, [allAssessments]);

  const runJudgeConfiguration = useModelTraceExplorerRunJudgesContext();

  return (
    <div
      data-testid="assessments-pane"
      css={{
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.md,
        paddingTop: theme.spacing.sm,
        height: '100%',
        borderLeft: `1px solid ${theme.colors.border}`,
        overflowY: 'auto',
        minWidth: ASSESSMENT_PANE_MIN_WIDTH,
        width: '100%',
        boxSizing: 'border-box',
      }}
      className={className}
    >
      <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        {assessmentsTitleOverride ? (
          assessmentsTitleOverride()
        ) : (
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Assessments" description="Title for the assessments pane" />
          </Typography.Title>
        )}
        {!disableCloseButton && (
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
      <hr css={{ border: 'none', borderTop: `1px solid ${theme.colors.border}`, margin: `${theme.spacing.xs}px 0` }} />
      <AssessmentsPaneFeedbackSection
        enableRunScorer={
          enableRunScorer &&
          isEvaluatingTracesInDetailsViewEnabled() &&
          Boolean(runJudgeConfiguration.renderRunJudgeModal)
        }
        feedbacks={feedbacks}
        activeSpanId={activeSpanId}
        traceId={traceId}
        sessionId={sessionId}
      />
      <Spacer size="sm" shrinks={false} />
      <AssessmentsPaneExpectationsSection
        expectations={expectations}
        activeSpanId={activeSpanId}
        traceId={traceId}
        sessionId={sessionId}
      />
      {issues.length > 0 && (
        <>
          <Spacer size="sm" shrinks={false} />
          <AssessmentsPaneIssuesSection issues={issues} selectedIssueId={selectedIssueId} />
        </>
      )}
    </div>
  );
};
