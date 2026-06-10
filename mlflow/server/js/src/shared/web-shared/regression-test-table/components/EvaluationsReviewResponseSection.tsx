import { Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { EvaluationsReviewTextBox } from './EvaluationsReviewTextBox';
import {
  isRetrievedContext,
  KnownEvaluationResultAssessmentOutputLabel,
  KnownEvaluationResultAssessmentTargetLabel,
} from './GenAiEvaluationTracesReview.utils';
import { VerticalBar } from './VerticalBar';
import type { RunEvaluationTracesDataEntry } from '../types';

const EvaluationsReviewSingleRunResponseSection = ({
  evaluationResult,
}: {
  evaluationResult: RunEvaluationTracesDataEntry;
}) => {
  const { theme } = useDesignSystemTheme();

  const { outputs, targets } = evaluationResult;

  // Filter out retrieve_context values
  const outputEntries = Object.entries(outputs).filter(([, value]) => !isRetrievedContext(value));
  const targetEntries = Object.entries(targets).filter(([, value]) => !isRetrievedContext(value));

  const noValues = outputEntries.length === 0 && targetEntries.length === 0;

  return (
    <div css={{ paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md, width: '100%' }}>
      <Typography.Text bold>
        <FormattedMessage defaultMessage="Response" description="Evaluation review > Response section > title" />
      </Typography.Text>
      <Spacer size="sm" />
      {noValues && (
        <Typography.Paragraph>
          <FormattedMessage
            defaultMessage="No responses or targets logged"
            description="Evaluation review > Response section > no values"
          />
        </Typography.Paragraph>
      )}
      <div css={{ display: 'flex', gap: theme.spacing.md, alignItems: 'flex-start' }}>
        {outputEntries.length > 0 && (
          <div css={{ flex: 1 }}>
            {outputEntries.map(([key, output]) => {
              const mappedTitle = KnownEvaluationResultAssessmentOutputLabel[key];
              const title = mappedTitle ? <FormattedMessage {...mappedTitle} /> : key;
              return <EvaluationsReviewTextBox key={key} fieldName={key} title={title} value={output} showCopyIcon />;
            })}
          </div>
        )}

        {targetEntries.length > 0 && (
          <div css={{ flex: 1 }}>
            {targetEntries.map(([key, output]) => {
              const mappedTitle = KnownEvaluationResultAssessmentTargetLabel[key];
              const title = mappedTitle ? <FormattedMessage {...mappedTitle} /> : key;
              return <EvaluationsReviewTextBox key={key} fieldName={key} title={title} value={output} showCopyIcon />;
            })}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Displays responses and targets for a given evaluation result.
 */
export const EvaluationsReviewResponseSection = ({
  evaluationResult,
  otherEvaluationResult,
}: {
  evaluationResult?: RunEvaluationTracesDataEntry;
  otherEvaluationResult?: RunEvaluationTracesDataEntry;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        width: '100%',
        gap: theme.spacing.sm,
      }}
    >
      {evaluationResult && <EvaluationsReviewSingleRunResponseSection evaluationResult={evaluationResult} />}
      {otherEvaluationResult && (
        <>
          <VerticalBar />
          <EvaluationsReviewSingleRunResponseSection evaluationResult={otherEvaluationResult} />
        </>
      )}
    </div>
  );
};
