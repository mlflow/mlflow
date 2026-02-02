import { Tag, Tooltip, Typography } from '@databricks/design-system';
import { aggregateNumericAssessments } from '../utils/SessionAggregationUtils';
import { AssessmentInfo } from '../types';
import { FormattedMessage } from '@databricks/i18n';
import { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { NullCell } from './NullCell';

const getDisplayValue = (average: number) => {
  const formattedAverage = Number.isInteger(average) ? average.toString() : average.toFixed(2).replace(/\.?0+$/, '');
  return (
    <FormattedMessage
      defaultMessage="{formattedAverage} (AVG)"
      description="Label showing the average value of numeric assessments"
      values={{ formattedAverage }}
    />
  );
};

export const SessionHeaderNumericAggregatedCell = ({
  assessmentInfo,
  traces,
}: {
  assessmentInfo: AssessmentInfo;
  traces: ModelTraceInfoV3[];
}) => {
  const { average, count } = aggregateNumericAssessments(traces, assessmentInfo.name);

  if (average === null) {
    return <NullCell />;
  }

  return (
    <Tooltip
      componentId="mlflow.genai-traces-table.session-numeric-assessment"
      content={
        <FormattedMessage
          defaultMessage="Average of {count} values"
          description="Tooltip for numeric assessment average in session header"
          values={{ count }}
        />
      }
    >
      <Tag componentId="mlflow.genai-traces-table.average-values-tag">
        <Typography.Text size="sm">{getDisplayValue(average)}</Typography.Text>
      </Tag>
    </Tooltip>
  );
};
