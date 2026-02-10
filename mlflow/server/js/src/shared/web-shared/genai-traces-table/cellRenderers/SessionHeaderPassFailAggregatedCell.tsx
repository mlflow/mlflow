import { DangerIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
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
  const { passCount, totalCount, errorCount } = aggregatePassFailAssessments(traces, assessmentInfo);

  if (totalCount > 0 || errorCount > 0) {
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
          gap: theme.spacing.xs,
          minWidth: 0,
          width: totalCount > 0 ? '100%' : 'fit-content',
          cursor: onExpandSession ? 'pointer' : undefined,
          borderRadius: theme.borders.borderRadiusSm,
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
        {totalCount > 0 && (
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
        )}
        <span
          css={{
            display: 'flex',
            alignItems: 'center',
            flexShrink: 0,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textPrimary,
            whiteSpace: 'nowrap',
            width: 50,
          }}
        >
          {totalCount > 0 ? (
            <>
              <Typography.Text bold size="sm">
                {passCount}/{totalCount}
              </Typography.Text>
              {errorCount > 0 && (
                <DangerIcon css={{ fontSize: 12, marginLeft: 4, color: theme.colors.textValidationWarning }} />
              )}
            </>
          ) : (
            <span css={{ display: 'flex', alignItems: 'center', color: theme.colors.textValidationWarning }}>
              <DangerIcon css={{ fontSize: 12, marginRight: 4 }} />
              <Typography.Text size="sm" color="warning">
                <FormattedMessage defaultMessage="Error" description="Label shown when all assessments have errors" />
              </Typography.Text>
            </span>
          )}
        </span>
      </div>
    );

    const getTooltipContent = () => {
      const expandMessage = intl.formatMessage({
        defaultMessage: 'Click to expand',
        description: 'Tooltip for expandable session cell',
      });

      if (errorCount > 0) {
        const errorMessage = intl.formatMessage(
          {
            defaultMessage: '{errorCount, plural, one {# error} other {# errors}}',
            description: 'Error count in tooltip',
          },
          { errorCount },
        );
        return `${errorMessage} · ${expandMessage}`;
      }

      return expandMessage;
    };

    if (onExpandSession) {
      return (
        <Tooltip
          componentId="mlflow.genai-traces-table.session-header.pass-fail-aggregated"
          content={getTooltipContent()}
        >
          {content}
        </Tooltip>
      );
    }

    if (errorCount > 0) {
      return (
        <Tooltip
          componentId="mlflow.genai-traces-table.session-header.error-count"
          content={intl.formatMessage(
            {
              defaultMessage: '{errorCount, plural, one {# error} other {# errors}}',
              description: 'Tooltip showing error count',
            },
            { errorCount },
          )}
        >
          {content}
        </Tooltip>
      );
    }

    return content;
  }

  return <NullCell />;
};
