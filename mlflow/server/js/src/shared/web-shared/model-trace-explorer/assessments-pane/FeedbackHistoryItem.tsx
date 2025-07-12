import { isNil } from 'lodash';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';

import { AssessmentDisplayValue } from './AssessmentDisplayValue';
import { AssessmentItemHeader } from './AssessmentItemHeader';
import { FeedbackErrorItem } from './FeedbackErrorItem';
import type { FeedbackAssessment } from '../ModelTrace.types';

// this is mostly a copy of FeedbackItem, but with
// different styling and no ability to edit.
export const FeedbackHistoryItem = ({ feedback }: { feedback: FeedbackAssessment }) => {
  const { theme } = useDesignSystemTheme();
  const value = feedback.feedback.value;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
      }}
    >
      <AssessmentItemHeader renderConnector={false} assessment={feedback} />
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          marginLeft: 10,
          paddingLeft: theme.spacing.md,
          paddingTop: theme.spacing.sm,
          paddingBottom: theme.spacing.md,
          paddingRight: theme.spacing.lg,
          borderLeft: `1px solid ${theme.colors.border}`,
        }}
      >
        {!isNil(feedback.feedback.error) ? (
          <FeedbackErrorItem error={feedback.feedback.error} />
        ) : (
          <>
            <Typography.Text size="sm" color="secondary">
              <FormattedMessage defaultMessage="Feedback" description="Label for the value of an feedback assessment" />
            </Typography.Text>
            <div>
              <AssessmentDisplayValue jsonValue={JSON.stringify(value)} />
            </div>
          </>
        )}
        {feedback.rationale && (
          <>
            <Typography.Text size="sm" color="secondary" css={{ marginTop: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Rationale"
                description="Label for the rationale of an expectation assessment"
              />
            </Typography.Text>
            <GenAIMarkdownRenderer>{feedback.rationale}</GenAIMarkdownRenderer>
          </>
        )}
      </div>
    </div>
  );
};
