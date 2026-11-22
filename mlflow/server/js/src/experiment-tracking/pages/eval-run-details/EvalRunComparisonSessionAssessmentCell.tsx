import { Tag, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceTableColumn } from '@databricks/web-shared/traces-table';
import { AssessmentCell } from '../../components/experiment-page/components/traces-v4/components/TraceAssessmentCell';
import { assessmentNameFromColumnId } from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import {
  getSessionAssessmentSummary,
  pickComparisonFeedback,
  type SessionComparisonAssessmentInfo,
} from './evalRunComparisonSessionSummaries';

export interface EvalRunComparisonSessionAssessmentCellProps {
  traces: ModelTraceInfoV3[];
  info: SessionComparisonAssessmentInfo;
}

export const EvalRunComparisonSessionAssessmentCell = ({
  traces,
  info,
}: EvalRunComparisonSessionAssessmentCellProps) => {
  const intl = useIntl();
  const summary = getSessionAssessmentSummary(traces, info);
  if (info.scope === 'session') {
    const assessment = summary.assessments[0];
    return assessment && (summary.values.length > 0 || summary.hasError) ? (
      <AssessmentCell assessment={assessment} columnType={info.dtype === 'numeric' ? 'numeric' : 'categorical'} />
    ) : (
      <Typography.Text color="secondary">—</Typography.Text>
    );
  }
  if (summary.value !== undefined) {
    const isRate = info.dtype === 'boolean' || info.dtype === 'pass-fail';
    return (
      <Tag componentId="mlflow.eval-run-comparison.session-assessment-summary">
        {intl.formatMessage(
          { defaultMessage: '{value} ({label})', description: 'Trace feedback summary within one comparison session' },
          {
            value: intl.formatNumber(
              summary.value,
              isRate ? { style: 'percent', maximumFractionDigits: 0 } : { maximumFractionDigits: 2 },
            ),
            label:
              info.dtype === 'boolean'
                ? intl.formatMessage({
                    defaultMessage: 'TRUE',
                    description: 'True-rate label in a comparison session score summary',
                  })
                : info.dtype === 'pass-fail'
                  ? intl.formatMessage({
                      defaultMessage: 'PASS',
                      description: 'Pass-rate label in a comparison session score summary',
                    })
                  : intl.formatMessage({
                      defaultMessage: 'AVG',
                      description: 'Average label in a comparison session score summary',
                    }),
          },
        )}
      </Tag>
    );
  }
  if (summary.values.length) {
    const counts = new Map<string, number>();
    for (const value of summary.values) {
      const label = typeof value === 'string' ? value : JSON.stringify(value);
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return (
      <span>
        {[...counts].map(([value, count]) => (
          <Tag key={value} componentId="mlflow.eval-run-comparison.session-assessment-category">
            <FormattedMessage
              defaultMessage="{value}: {count}"
              description="Categorical feedback counts within a session"
              values={{ value, count }}
            />
          </Tag>
        ))}
      </span>
    );
  }
  const failed = summary.assessments.find((assessment) => assessment.error || assessment.feedback.error);
  return failed ? (
    <AssessmentCell assessment={failed} columnType="categorical" />
  ) : (
    <Typography.Text color="secondary">—</Typography.Text>
  );
};

export const buildComparisonSessionAssessmentColumns = (
  columns: TraceTableColumn[],
  infos: SessionComparisonAssessmentInfo[],
): TraceTableColumn[] =>
  columns.map((column) => {
    const info = infos.find((candidate) => candidate.name === assessmentNameFromColumnId(column.id ?? ''));
    if (!info) {
      return column;
    }
    return {
      ...column,
      cell: (context) => {
        if (info.scope === 'session') {
          return null;
        }
        const assessment = pickComparisonFeedback([context.row.original], info.name, 'trace');
        return assessment ? (
          <AssessmentCell assessment={assessment} columnType={info.dtype === 'numeric' ? 'numeric' : 'categorical'} />
        ) : null;
      },
      renderSessionCell: (traces) => <EvalRunComparisonSessionAssessmentCell traces={traces} info={info} />,
    };
  });
