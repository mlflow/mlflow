import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { AssessmentDisplayValue } from './assessments-pane/AssessmentDisplayValue';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const omitNullFields = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(omitNullFields);
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, field]) => field !== null && field !== undefined)
        .map(([key, field]) => [key, omitNullFields(field)]),
    );
  }
  return value;
};

export const getEvaluatorOutputFields = (outputs: unknown): { key: string; value: string }[] => {
  if (!isRecord(outputs)) {
    return outputs == null ? [] : [{ key: '', value: JSON.stringify(outputs, null, 2) }];
  }

  const fields = omitNullFields(outputs) as Record<string, unknown>;
  if (isRecord(fields.feedback) && 'value' in fields.feedback) {
    fields.feedback = fields.feedback.value;
  }
  if (!('feedback' in fields) && 'value' in fields) {
    fields.feedback = fields.value;
    delete fields.value;
  }

  return Object.entries(fields)
    .sort(([first], [second]) => {
      const priority = (key: string) => (key === 'feedback' ? 0 : key === 'rationale' ? 1 : 2);
      return priority(first) - priority(second);
    })
    .map(([key, value]) => ({ key, value: JSON.stringify(value, null, 2) }));
};

export const EvaluatorFeedbackField = ({ value, assessmentName }: { value: string; assessmentName: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text color="secondary" size="sm">
        <FormattedMessage defaultMessage="feedback" description="Feedback field of an evaluator span" />
      </Typography.Text>
      <div css={{ alignSelf: 'flex-start', maxWidth: '100%' }}>
        <AssessmentDisplayValue jsonValue={value} assessmentName={assessmentName} />
      </div>
    </div>
  );
};
