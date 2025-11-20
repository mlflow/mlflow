import { isNil } from 'lodash';
import { useMemo } from 'react';

import { GavelIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { Assessment, AssessmentError, AssessmentMetadata, FeedbackAssessment } from '../ModelTrace.types';
import { AssessmentDisplayValue } from '../assessments-pane/AssessmentDisplayValue';
import { FeedbackErrorItem } from '../assessments-pane/FeedbackErrorItem';
import { getAssessmentValue } from '../assessments-pane/utils';

export const SIMPLIFIED_ASSESSMENT_VIEW_MIN_WIDTH = 300;

const MetadataDisplay = ({ metadata }: { metadata?: AssessmentMetadata }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const spanName = metadata?.span_name;

  if (!metadata || !spanName) {
    return null;
  }

  const spanLabel = intl.formatMessage(
    {
      defaultMessage: 'Span: {spanName}',
      description: 'Label for the span name in assessment metadata',
    },
    { spanName },
  );

  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
      <Tooltip componentId="shared.model-trace-explorer.span-name-tooltip" content={spanLabel}>
        <Tag
          css={{ display: 'inline-flex', maxWidth: '100%' }}
          componentId="shared.model-trace-explorer.span-name-tag"
          color="default"
        >
          <span
            css={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              textWrap: 'nowrap',
            }}
          >
            {spanLabel}
          </span>
        </Tag>
      </Tooltip>
    </div>
  );
};

const AssessmentCard = ({ assessment }: { assessment: FeedbackAssessment }) => {
  const { theme } = useDesignSystemTheme();
  const value = getAssessmentValue(assessment);
  const rationale = assessment.rationale;
  const hasError = !isNil(assessment.feedback.error);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.md,
        gap: theme.spacing.sm,
      }}
    >
      {/* Header with icon, title */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flex: 1, minWidth: 0 }}>
          <GavelIcon css={{ flexShrink: 0 }} />
          <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
            {assessment.assessment_name}
          </Typography.Text>
        </div>
      </div>

      {/* Error display */}
      {hasError && <FeedbackErrorItem error={assessment.feedback.error as AssessmentError} />}

      {/* Result value */}
      {!hasError && value !== undefined && (
        <div css={{ display: 'flex', alignItems: 'center' }}>
          <AssessmentDisplayValue jsonValue={JSON.stringify(value)} />
        </div>
      )}

      {/* Metadata mini-cards (span name and document URI) */}
      <MetadataDisplay metadata={assessment.metadata as AssessmentMetadata | undefined} />

      {/* Rationale */}
      {rationale && (
        <Typography.Text css={{ color: theme.colors.textSecondary, lineHeight: theme.typography.lineHeightBase }}>
          {rationale}
        </Typography.Text>
      )}
    </div>
  );
};

export const SimplifiedAssessmentView = ({ assessments }: { assessments: Assessment[] }) => {
  const { theme } = useDesignSystemTheme();

  // We only show valid feedback assessments
  const feedbackAssessments: FeedbackAssessment[] = useMemo(
    () =>
      assessments.filter(
        (assessment) => 'feedback' in assessment && assessment.valid !== false,
      ) as FeedbackAssessment[],
    [assessments],
  );

  if (feedbackAssessments.length === 0) {
    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          padding: theme.spacing.md,
          minWidth: SIMPLIFIED_ASSESSMENT_VIEW_MIN_WIDTH,
          height: '100%',
          borderLeft: `1px solid ${theme.colors.border}`,
        }}
      >
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No assessments available"
            description="Message shown when there are no assessments to display in simplified view"
          />
        </Typography.Text>
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.md,
        paddingTop: theme.spacing.sm,
        gap: theme.spacing.md,
        minWidth: SIMPLIFIED_ASSESSMENT_VIEW_MIN_WIDTH,
        height: '100%',
        borderLeft: `1px solid ${theme.colors.border}`,
        overflowY: 'auto',
      }}
    >
      {feedbackAssessments.map((assessment) => (
        <AssessmentCard key={assessment.assessment_id} assessment={assessment} />
      ))}
    </div>
  );
};
