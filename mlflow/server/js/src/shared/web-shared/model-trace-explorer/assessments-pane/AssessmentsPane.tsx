import { partition, uniqBy } from 'lodash';
import { useMemo } from 'react';

import { Button, CloseIcon, Spacer, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ASSESSMENT_PANE_MIN_WIDTH } from './AssessmentsPane.utils';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment } from '../ModelTrace.types';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useTraceCachedActions } from '../hooks/useTraceCachedActions';
import { AssessmentsPaneExpectationsSection } from './AssessmentsPaneExpectationsSection';
import { AssessmentsPaneFeedbackSection } from './AssessmentsPaneFeedbackSection';

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
  const { setAssessmentsPaneExpanded, isInComparisonView } = useModelTraceExplorerViewState();
  const [feedbacks, expectations] = useMemo(
    () => partition(allAssessments, (assessment) => 'feedback' in assessment),
    [allAssessments],
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
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: theme.spacing.sm,
          marginBottom: theme.spacing.sm,
        }}
      >
        {!isInComparisonView &&
          (assessmentsTitleOverride ? (
            assessmentsTitleOverride()
          ) : (
            <Typography.Title level={3} withoutMargins css={{ flexShrink: 0 }}>
              <FormattedMessage defaultMessage="Assessments" description="Label for the assessments pane" />
            </Typography.Title>
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
      <AssessmentsPaneFeedbackSection feedbacks={feedbacks} activeSpanId={activeSpanId} traceId={traceId} />
      <Spacer size="sm" shrinks={false} />
      <AssessmentsPaneExpectationsSection expectations={expectations} activeSpanId={activeSpanId} traceId={traceId} />
    </div>
  );
};
