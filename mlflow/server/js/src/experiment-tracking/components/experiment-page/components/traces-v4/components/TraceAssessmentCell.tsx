import { HoverCard, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, FormattedRelativeTime } from '@databricks/i18n';
import {
  getAssessmentValue,
  type Assessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
// These three aren't re-exported from the model-trace-explorer barrel, so import them from their
// subpaths rather than editing the generated index.ts.
import { AssessmentDisplayValue } from '@databricks/web-shared/model-trace-explorer/assessments-pane/AssessmentDisplayValue';
import { AssessmentSourceName } from '@databricks/web-shared/model-trace-explorer/assessments-pane/AssessmentSourceName';
import { getSourceIcon } from '@databricks/web-shared/model-trace-explorer/assessments-pane/utils';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';
import { pickCellAssessment } from '../utils/assessmentColumns';

export interface TraceAssessmentCellProps {
  trace: ModelTraceInfoV3;
  assessmentName: string;
}

/**
 * Hover-card body for an assessment cell: the value tag, an optional rationale (markdown), the
 * source (icon + id), and when it was last updated. Extracted to module scope so it can be unit
 * tested directly without waiting on HoverCard's open delay.
 */
export const TraceAssessmentHoverContent = ({ assessment }: { assessment: Assessment }) => {
  const { theme } = useDesignSystemTheme();
  const SourceIcon = getSourceIcon(assessment.source);
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <AssessmentDisplayValue
        jsonValue={getAssessmentValue(assessment)?.toString() ?? ''}
        assessmentName={assessment.assessment_name}
      />
      {assessment.rationale && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="Rationale"
              description="Label for the rationale section of an assessment hover card"
            />
          </Typography.Text>
          <div css={{ '& > div:last-of-type': { marginBottom: 0 } }}>
            <GenAIMarkdownRenderer compact>{assessment.rationale}</GenAIMarkdownRenderer>
          </div>
        </div>
      )}
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Source"
            description="Label for the source section of an assessment hover card"
          />
        </Typography.Text>
        <div css={{ display: 'flex', alignItems: 'center' }}>
          <SourceIcon size={theme.typography.fontSizeSm} />
          <AssessmentSourceName source={assessment.source} />
        </div>
      </div>
      {assessment.last_update_time && (
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Updated {time}"
            description="Relative last-updated time on an assessment hover card"
            values={{
              time: (
                <FormattedRelativeTime
                  value={(new Date(assessment.last_update_time).getTime() - Date.now()) / 1000}
                  numeric="auto"
                  updateIntervalInSeconds={60}
                />
              ),
            }}
          />
        </Typography.Text>
      )}
    </div>
  );
};

/**
 * Renders one trace's value for a single assessment column: the most recent assessment of that name
 * as a colored tag (mirrors the prior tab), wrapped in a hover card that reveals rationale, source,
 * and last-updated time. Renders nothing when the trace has no such assessment.
 */
export const TraceAssessmentCell = ({ trace, assessmentName }: TraceAssessmentCellProps) => {
  const assessment = pickCellAssessment(trace, assessmentName);
  if (!assessment) {
    return null;
  }
  return (
    <HoverCard
      trigger={
        <span>
          <AssessmentDisplayValue
            jsonValue={getAssessmentValue(assessment)?.toString() ?? ''}
            assessmentName={assessmentName}
            // The hover card already reveals the full value, so the tag's own tooltip would be redundant.
            disableTooltip
          />
        </span>
      }
      content={<TraceAssessmentHoverContent assessment={assessment} />}
    />
  );
};
