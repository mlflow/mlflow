import { HoverCard, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { aggregateStringAssessments } from '../utils/SessionAggregationUtils';
import { AssessmentInfo } from '../types';
import { FormattedMessage } from '@databricks/i18n';
import { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { NullCell } from './NullCell';

export const SessionHeaderStringAggregatedCell = ({
  assessmentInfo,
  traces,
}: {
  assessmentInfo: AssessmentInfo;
  traces: ModelTraceInfoV3[];
}) => {
  const { valueCounts, totalCount } = aggregateStringAssessments(traces, assessmentInfo.name);
  const { theme } = useDesignSystemTheme();
  const uniqueCount = valueCounts.size;

  // Sort values by count (descending)
  const sortedEntries = Array.from(valueCounts.entries()).sort((a, b) => b[1] - a[1]);

  if (totalCount === 0) {
    return <NullCell />;
  }

  return (
    <HoverCard
      trigger={
        <Tag css={{ cursor: 'default' }} componentId="mlflow.genai-traces-table.session-string-tag">
          <FormattedMessage
            defaultMessage="{count} {count, plural, one {value} other {values}}"
            description="Tag showing the number of unique string values in session assessment"
            values={{ count: uniqueCount }}
          />
        </Tag>
      }
      content={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {sortedEntries.map(([value, count]) => (
            <div
              key={value}
              css={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: theme.spacing.md,
              }}
            >
              <Typography.Text ellipsis css={{ maxWidth: 200 }}>
                {value}
              </Typography.Text>
              <Typography.Text color="secondary">{count}</Typography.Text>
            </div>
          ))}
        </div>
      }
    />
  );
};
