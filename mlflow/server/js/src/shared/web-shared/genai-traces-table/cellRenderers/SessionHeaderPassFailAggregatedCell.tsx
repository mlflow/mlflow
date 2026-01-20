import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { AssessmentInfo } from '../types';
import { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { aggregatePassFailAssessments } from '../utils/SessionAggregationUtils';
import { FAIL_BARCHART_BAR_COLOR, PASS_BARCHART_BAR_COLOR } from '../utils/Colors';
import { NullCell } from './NullCell';

export const SessionHeaderPassFailAggregatedCell = ({
  assessmentInfo,
  traces,
}: {
  assessmentInfo: AssessmentInfo;
  traces: ModelTraceInfoV3[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Non-session-level assessment column - aggregate values from all traces
  const { passCount, totalCount } = aggregatePassFailAssessments(traces, assessmentInfo);

  if (totalCount > 0) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          minWidth: 0,
          width: '100%',
        }}
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
  }

  return <NullCell />;
};
