import { ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { DecisionQuestionViewModel } from './Decision.types';

const formatDisplayValue = (value: unknown): string => {
  if (typeof value === 'string') {
    return value;
  }

  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
};

const SerializedValue = ({ value }: { value: unknown }) => {
  const { theme } = useDesignSystemTheme();
  const isStructuredValue = typeof value === 'object' && value !== null;

  if (!isStructuredValue) {
    return (
      <Typography.Text css={{ overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}>
        {formatDisplayValue(value)}
      </Typography.Text>
    );
  }

  return (
    <pre
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusSm,
        boxSizing: 'border-box',
        color: theme.colors.textPrimary,
        fontFamily: 'monospace',
        fontSize: theme.typography.fontSizeSm,
        lineHeight: theme.typography.lineHeightBase,
        margin: 0,
        maxWidth: '100%',
        overflowX: 'auto',
        padding: theme.spacing.sm,
        whiteSpace: 'pre',
      }}
    >
      {formatDisplayValue(value)}
    </pre>
  );
};

const QuestionType = ({ declaredType }: { declaredType?: string }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Typography.Text
      color="secondary"
      size="sm"
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusSm,
        flexShrink: 0,
        padding: `0 ${theme.spacing.xs}`,
      }}
    >
      {declaredType ?? <FormattedMessage defaultMessage="Unknown type" description="Unknown decision question type" />}
    </Typography.Text>
  );
};

const QuestionSummary = ({ question, expandable }: { question: DecisionQuestionViewModel; expandable: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const instructionPreview =
    typeof question.instructions === 'string' && question.instructions.trim() ? question.instructions : undefined;

  return (
    <span
      css={{
        alignItems: 'center',
        display: 'flex',
        gap: theme.spacing.md,
        justifyContent: 'space-between',
        minWidth: 0,
        padding: theme.spacing.md,
      }}
    >
      <span css={{ display: 'flex', flex: 1, flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
        <span css={{ alignItems: 'baseline', display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
          <Typography.Text bold css={{ overflowWrap: 'anywhere' }}>
            {question.id}
          </Typography.Text>
          <QuestionType declaredType={question.declaredType} />
        </span>
        {instructionPreview && (
          <Typography.Text
            color="secondary"
            size="sm"
            css={{
              display: '-webkit-box',
              overflow: 'hidden',
              overflowWrap: 'anywhere',
              whiteSpace: 'pre-wrap',
              WebkitBoxOrient: 'vertical',
              WebkitLineClamp: 2,
            }}
          >
            {instructionPreview}
          </Typography.Text>
        )}
      </span>
      {expandable && (
        <span
          aria-hidden="true"
          css={{
            color: theme.colors.textSecondary,
            display: 'flex',
            flexShrink: 0,
            transition: 'transform 150ms ease',
            'details[open] &': {
              transform: 'rotate(90deg)',
            },
          }}
        >
          <ChevronRightIcon />
        </span>
      )}
    </span>
  );
};

const QuestionDetails = ({ question }: { question: DecisionQuestionViewModel }) => {
  const { theme } = useDesignSystemTheme();
  const hasStructuredInstructions = question.instructions !== undefined && typeof question.instructions !== 'string';

  return (
    <div
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        borderTop: `1px solid ${theme.colors.borderDecorative}`,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
      }}
    >
      {hasStructuredInstructions && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold size="sm">
            <FormattedMessage defaultMessage="Instructions" description="Label for decision question instructions" />
          </Typography.Text>
          <SerializedValue value={question.instructions} />
        </div>
      )}
      {question.criteria !== undefined && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold size="sm">
            <FormattedMessage defaultMessage="Criteria" description="Label for decision question criteria" />
          </Typography.Text>
          <SerializedValue value={question.criteria} />
        </div>
      )}
    </div>
  );
};

const QuestionRow = ({ question }: { question: DecisionQuestionViewModel }) => {
  const { theme } = useDesignSystemTheme();
  const isExpandable =
    (question.instructions !== undefined && typeof question.instructions !== 'string') ||
    question.criteria !== undefined;

  if (!isExpandable) {
    return (
      <div role="listitem">
        <QuestionSummary question={question} expandable={false} />
      </div>
    );
  }

  return (
    <div role="listitem">
      <details>
        <summary
          css={{
            cursor: 'pointer',
            listStyle: 'none',
            '&::-webkit-details-marker': { display: 'none' },
            '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
            '&:active': { backgroundColor: theme.colors.actionDefaultBackgroundPress },
            '&:focus-visible': {
              outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
              outlineOffset: -2,
            },
          }}
        >
          <QuestionSummary question={question} expandable />
        </summary>
        <QuestionDetails question={question} />
      </details>
    </div>
  );
};

export const DecisionQuestionsRenderer = ({
  questions,
}: {
  questions: readonly DecisionQuestionViewModel[];
}): React.ReactElement | null => {
  const { theme } = useDesignSystemTheme();

  if (questions.length === 0) {
    return null;
  }

  return (
    <div
      role="list"
      data-testid="decision-questions"
      css={{
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusMd,
        containerType: 'inline-size',
        overflow: 'hidden',
        '& > [role="listitem"] + [role="listitem"]': {
          borderTop: `1px solid ${theme.colors.borderDecorative}`,
        },
      }}
    >
      {questions.map((question, index) => (
        <QuestionRow key={`${question.id}-${index}`} question={question} />
      ))}
    </div>
  );
};
