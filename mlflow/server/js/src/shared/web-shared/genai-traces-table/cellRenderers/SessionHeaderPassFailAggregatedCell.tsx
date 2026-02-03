import { Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { AssessmentInfo } from '../types';
import { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { aggregatePassFailAssessments } from '../utils/SessionAggregationUtils';
import { FAIL_BARCHART_BAR_COLOR, PASS_BARCHART_BAR_COLOR } from '../utils/Colors';
import { NullCell } from './NullCell';

export const SessionHeaderPassFailAggregatedCell = ({
  assessmentInfo,
  traces,
  onExpandSession,
}: {
  assessmentInfo: AssessmentInfo;
  traces: ModelTraceInfoV3[];
  onExpandSession?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Non-session-level assessment column - aggregate values from all traces
  const { passCount, totalCount } = aggregatePassFailAssessments(traces, assessmentInfo);

  if (totalCount > 0) {
    const handleClick = () => {
      onExpandSession?.();
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onExpandSession?.();
      }
    };

    const content = (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          minWidth: 0,
          width: '100%',
          cursor: onExpandSession ? 'pointer' : undefined,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.xs,
          margin: -theme.spacing.xs,
          '&:hover': onExpandSession
            ? {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover,
              }
            : undefined,
        }}
        onClick={onExpandSession ? handleClick : undefined}
        onKeyDown={onExpandSession ? handleKeyDown : undefined}
        role={onExpandSession ? 'button' : undefined}
        tabIndex={onExpandSession ? 0 : undefined}
      >
        <div
          css={{
            display: 'flex',
            flex: 1,
            minWidth: 0,
            height: theme.spacing.sm,
            borderRadius: theme.borders.borderRadiusMd,
            overflow: 'hidden',
          }}
        >
          {passCount > 0 && (
            <div
              css={{
                flex: passCount,
                backgroundColor: PASS_BARCHART_BAR_COLOR,
              }}
            />
          )}
          {passCount < totalCount && (
            <div
              css={{
                flex: totalCount - passCount,
                backgroundColor: FAIL_BARCHART_BAR_COLOR,
              }}
            />
          )}
        </div>
        <span
          css={{
            flexShrink: 0,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textPrimary,
            whiteSpace: 'nowrap',
          }}
        >
          <Typography.Text css={{ marginRight: theme.spacing.xs }} bold size="sm">
            {passCount}/{totalCount}
          </Typography.Text>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="PASS"
              description="Label for an aggregate display showing how many assessments have passed or failed"
            />
          </Typography.Text>
        </span>
      </div>
    );

    if (onExpandSession) {
      return (
        <Tooltip
          componentId="mlflow.genai-traces-table.session-header.pass-fail-aggregated"
          content={intl.formatMessage({
            defaultMessage: 'Click to expand session and see individual trace details',
            description: 'Tooltip for pass/fail aggregated cell in session header',
          })}
        >
          {content}
        </Tooltip>
      );
    }

    return content;
  }

  return <NullCell />;
};
