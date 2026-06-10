import { Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { EvaluationsReviewTextBox } from './EvaluationsReviewTextBox';
import type { RunEvaluationTracesDataEntry } from '../types';

/**
 * Displays inputs for a given evaluation result of a single run.
 */
const EvaluationsReviewSingleRunInputSection = ({
  evaluationResult,
}: {
  evaluationResult: RunEvaluationTracesDataEntry;
}) => {
  const { theme } = useDesignSystemTheme();
  const { inputs } = evaluationResult;
  const inputsEntries = Object.entries(inputs);
  const noValues = inputsEntries.length === 0;
  return (
    <div css={{ width: '100%', paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md }}>
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="{count, plural, one {Input} other {Inputs}}"
          description="Evaluation review > input section > title"
          values={{ count: inputsEntries.length }}
        />
      </Typography.Text>
      <Spacer size="sm" />
      {noValues && (
        <Typography.Paragraph>
          <FormattedMessage
            defaultMessage="No inputs logged"
            description="Evaluation review > input section > no values"
          />
        </Typography.Paragraph>
      )}
      {inputsEntries.map(([key, input]) => (
        <EvaluationsReviewTextBox fieldName={key} title={key} value={input} key={key} />
      ))}
    </div>
  );
};

/**
 * Displays inputs for a given evaluation result, across one or two runs.
 */
export const EvaluationsReviewInputSection = ({
  evaluationResult,
  otherEvaluationResult,
}: {
  evaluationResult?: RunEvaluationTracesDataEntry;
  otherEvaluationResult?: RunEvaluationTracesDataEntry;
}) => {
  const { theme } = useDesignSystemTheme();
  const inputsAreTheSame = evaluationResult?.inputsId === otherEvaluationResult?.inputsId;
  return (
    <div
      css={{
        display: 'flex',
        width: '100%',
        gap: theme.spacing.sm,
      }}
    >
      {evaluationResult && <EvaluationsReviewSingleRunInputSection evaluationResult={evaluationResult} />}
      {!inputsAreTheSame && otherEvaluationResult && (
        <EvaluationsReviewSingleRunInputSection evaluationResult={otherEvaluationResult} />
      )}
    </div>
  );
};
