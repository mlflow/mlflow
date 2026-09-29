import { useState } from 'react';

import {
  ChevronDownIcon,
  ChevronRightIcon,
  Progress,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type {
  DecisionAnswerViewModel,
  DecisionChoiceAnswerViewModel,
  DecisionNoulAnswerViewModel,
  DecisionScoreAnswerViewModel,
  DecisionUnknownAnswerViewModel,
} from './Decision.types';

let answerDisclosureIdCounter = 0;
const useAnswerDisclosureId = (): string => useState(() => `decision-answer-${++answerDisclosureIdCounter}`)[0];

const formatDisplayValue = (value: unknown): string => {
  if (typeof value === 'string') {
    return value;
  }
  if (value === undefined) {
    return '';
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
        backgroundColor: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusSm,
        color: theme.colors.textPrimary,
        fontFamily: 'monospace',
        fontSize: theme.typography.fontSizeSm,
        lineHeight: theme.typography.lineHeightBase,
        margin: 0,
        boxSizing: 'border-box',
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

const DetailField = ({ label, value }: { label: React.ReactNode; value: unknown }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
      <Typography.Text size="sm" bold>
        {label}
      </Typography.Text>
      <SerializedValue value={value} />
    </div>
  );
};

const ProbabilityRow = ({
  label,
  accessibleLabel,
  probability,
  emphasized = false,
}: {
  label: React.ReactNode;
  accessibleLabel: string;
  probability?: number;
  emphasized?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const formattedProbability =
    probability === undefined
      ? undefined
      : intl.formatNumber(probability, { style: 'percent', maximumFractionDigits: 1 });

  return (
    <div
      css={{
        alignItems: 'center',
        display: 'grid',
        gap: theme.spacing.sm,
        gridTemplateColumns: 'minmax(0, 1fr) minmax(64px, 2fr) min-content',
        minHeight: theme.general.heightSm,
        '@container (max-width: 420px)': {
          alignItems: 'baseline',
          gridTemplateColumns: 'minmax(0, 1fr) min-content',
        },
      }}
    >
      <div css={{ minWidth: 0, overflowWrap: 'anywhere' }}>{label}</div>
      <div
        css={{
          minWidth: 0,
          ...(probability === undefined && { gridColumn: '2 / -1' }),
          '@container (max-width: 420px)': {
            gridColumn: probability === undefined ? '2' : '1 / -1',
            gridRow: probability === undefined ? 1 : 2,
          },
        }}
      >
        {probability === undefined ? (
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="Not reported"
              description="Missing probability in a decision answer distribution"
            />
          </Typography.Text>
        ) : (
          <Progress.Root value={probability * 100} aria-label={accessibleLabel} aria-valuetext={formattedProbability}>
            <Progress.Indicator />
          </Progress.Root>
        )}
      </div>
      {formattedProbability && (
        <Typography.Text
          size="sm"
          color="secondary"
          bold={emphasized}
          css={{ '@container (max-width: 420px)': { gridColumn: 2, gridRow: 1 } }}
        >
          {formattedProbability}
        </Typography.Text>
      )}
    </div>
  );
};

const Distribution = ({ children }: { children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text size="sm" bold>
        <FormattedMessage
          defaultMessage="Probability distribution"
          description="Heading for a decision answer probability distribution"
        />
      </Typography.Text>
      <div css={{ display: 'flex', flexDirection: 'column' }}>{children}</div>
    </div>
  );
};

const Confidence = ({ value }: { value: number }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div
      css={{
        alignItems: 'baseline',
        borderTop: `1px solid ${theme.colors.borderDecorative}`,
        display: 'flex',
        flexWrap: 'wrap',
        gap: theme.spacing.sm,
        justifyContent: 'flex-end',
        paddingTop: theme.spacing.sm,
      }}
    >
      <Typography.Text size="sm" color="secondary">
        <FormattedMessage
          defaultMessage="Confidence"
          description="Label for the confidence reported by a decision answer"
        />
      </Typography.Text>
      <Typography.Text bold>{intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 1 })}</Typography.Text>
    </div>
  );
};

const ChoiceDetails = ({ answer }: { answer: DecisionChoiceAnswerViewModel }) => {
  const intl = useIntl();

  return (
    <>
      <Distribution>
        {answer.probabilities.map(({ label, probability }) => (
          <ProbabilityRow
            key={label}
            label={
              <Typography.Text size="sm" bold={label === answer.choice} css={{ overflowWrap: 'anywhere' }}>
                {label}
              </Typography.Text>
            }
            accessibleLabel={intl.formatMessage(
              {
                defaultMessage: 'Probability for {label}',
                description: 'Accessible label for a decision choice probability',
              },
              { label },
            )}
            probability={probability}
            emphasized={label === answer.choice}
          />
        ))}
      </Distribution>
      <Confidence value={answer.confidence} />
    </>
  );
};

const ScoreDetails = ({ answer }: { answer: DecisionScoreAnswerViewModel }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <>
      <Distribution>
        {answer.levels.map(({ score, description, probability }) => {
          const scoreLabel = intl.formatNumber(score);
          return (
            <ProbabilityRow
              key={score}
              label={
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
                  <Typography.Text size="sm" bold>
                    {scoreLabel}
                  </Typography.Text>
                  {description !== undefined && typeof description !== 'object' && (
                    <Typography.Text size="sm" color="secondary">
                      {formatDisplayValue(description)}
                    </Typography.Text>
                  )}
                  {description !== undefined && typeof description === 'object' && (
                    <SerializedValue value={description} />
                  )}
                </div>
              }
              accessibleLabel={intl.formatMessage(
                {
                  defaultMessage: 'Probability for score {score}',
                  description: 'Accessible label for a decision score probability',
                },
                { score: scoreLabel },
              )}
              probability={probability}
            />
          );
        })}
      </Distribution>
      <Confidence value={answer.confidence} />
    </>
  );
};

const NoulDetails = ({ answer }: { answer: DecisionNoulAnswerViewModel }) => {
  const intl = useIntl();
  const predictsTrue = answer.probabilityTrue >= 0.5;

  return (
    <Distribution>
      <ProbabilityRow
        label={
          <Typography.Text size="sm" bold={predictsTrue}>
            <FormattedMessage defaultMessage="True" description="True outcome in a decision Noul answer" />
          </Typography.Text>
        }
        accessibleLabel={intl.formatMessage({
          defaultMessage: 'Probability for true',
          description: 'Accessible label for the true probability in a decision Noul answer',
        })}
        probability={answer.probabilityTrue}
        emphasized={predictsTrue}
      />
      <ProbabilityRow
        label={
          <Typography.Text size="sm" bold={!predictsTrue}>
            <FormattedMessage defaultMessage="False" description="False outcome in a decision Noul answer" />
          </Typography.Text>
        }
        accessibleLabel={intl.formatMessage({
          defaultMessage: 'Probability for false',
          description: 'Accessible label for the false probability in a decision Noul answer',
        })}
        probability={1 - answer.probabilityTrue}
        emphasized={!predictsTrue}
      />
    </Distribution>
  );
};

const UnknownDetails = ({ answer }: { answer: DecisionUnknownAnswerViewModel }) => (
  <DetailField
    label={
      <FormattedMessage
        defaultMessage="Raw answer"
        description="Label for the raw data of an unrecognized decision answer"
      />
    }
    value={answer.rawAnswer}
  />
);

const AnswerDetails = ({ answer }: { answer: DecisionAnswerViewModel }) => {
  switch (answer.kind) {
    case 'choice':
      return <ChoiceDetails answer={answer} />;
    case 'score':
      return <ScoreDetails answer={answer} />;
    case 'noul':
      return <NoulDetails answer={answer} />;
    case 'unknown':
      return <UnknownDetails answer={answer} />;
  }
};

const AnswerKind = ({ answer }: { answer: DecisionAnswerViewModel }) => {
  if (answer.kind === 'unknown' && answer.declaredType) {
    return <>{answer.declaredType}</>;
  }

  switch (answer.kind) {
    case 'choice':
      return <FormattedMessage defaultMessage="Choice" description="Choice decision answer type" />;
    case 'score':
      return <FormattedMessage defaultMessage="Score" description="Score decision answer type" />;
    case 'noul':
      return <FormattedMessage defaultMessage="Noul" description="Noul decision answer type" />;
    case 'unknown':
      return <FormattedMessage defaultMessage="Unknown" description="Unknown decision answer type" />;
  }
};

const ConfidenceSummary = ({ value }: { value: number }) => {
  const intl = useIntl();

  return (
    <Typography.Text size="sm" color="secondary">
      <FormattedMessage
        defaultMessage="{confidence} confidence"
        description="Compact confidence summary for a decision answer"
        values={{ confidence: intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 1 }) }}
      />
    </Typography.Text>
  );
};

const ProbabilitySummary = ({ value }: { value: number }) => {
  const intl = useIntl();

  return (
    <Typography.Text size="sm" color="secondary">
      <FormattedMessage
        defaultMessage="{probability} probability"
        description="Compact probability summary for a decision Noul answer"
        values={{ probability: intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 1 }) }}
      />
    </Typography.Text>
  );
};

const AnswerSummary = ({ answer }: { answer: DecisionAnswerViewModel }) => {
  const intl = useIntl();

  if (answer.kind === 'choice') {
    return (
      <>
        <Typography.Text bold css={{ overflowWrap: 'anywhere' }}>
          {answer.choice}
        </Typography.Text>
        <ConfidenceSummary value={answer.confidence} />
      </>
    );
  }

  if (answer.kind === 'score') {
    return (
      <>
        <Typography.Text bold size="lg">
          {intl.formatNumber(answer.score, { maximumFractionDigits: 2 })}
        </Typography.Text>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Range {min}–{max}"
            description="Score range summary for a decision answer"
            values={{ min: intl.formatNumber(answer.range.min), max: intl.formatNumber(answer.range.max) }}
          />
        </Typography.Text>
        <ConfidenceSummary value={answer.confidence} />
      </>
    );
  }

  if (answer.kind === 'noul') {
    const predictsTrue = answer.probabilityTrue >= 0.5;
    const outcomeProbability = predictsTrue ? answer.probabilityTrue : 1 - answer.probabilityTrue;
    return (
      <>
        <Typography.Text bold css={{ overflowWrap: 'anywhere' }}>
          {predictsTrue ? (
            <FormattedMessage
              defaultMessage="true"
              description="Lowercase true outcome in a decision Noul answer summary"
            />
          ) : (
            <FormattedMessage
              defaultMessage="false"
              description="Lowercase false outcome in a decision Noul answer summary"
            />
          )}
        </Typography.Text>
        <ProbabilitySummary value={outcomeProbability} />
      </>
    );
  }

  return (
    <Typography.Text color="secondary">
      {answer.reason === 'malformed' ? (
        <FormattedMessage defaultMessage="Malformed answer" description="Malformed decision answer summary" />
      ) : (
        <FormattedMessage defaultMessage="Unsupported answer" description="Unsupported decision answer summary" />
      )}
    </Typography.Text>
  );
};

const AnswerRow = ({ answer }: { answer: DecisionAnswerViewModel }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const disclosureId = useAnswerDisclosureId();
  const buttonId = `${disclosureId}-button`;
  const detailsId = `${disclosureId}-details`;

  return (
    <div role="listitem">
      <button
        type="button"
        id={buttonId}
        aria-expanded={expanded}
        aria-controls={detailsId}
        onClick={() => setExpanded((value) => !value)}
        css={{
          alignItems: 'center',
          appearance: 'none',
          backgroundColor: theme.colors.backgroundPrimary,
          border: 0,
          color: theme.colors.textPrimary,
          display: 'grid',
          font: 'inherit',
          gap: theme.spacing.md,
          gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)',
          padding: theme.spacing.md,
          textAlign: 'left',
          width: '100%',
          '@container (max-width: 420px)': {
            alignItems: 'stretch',
            gap: theme.spacing.sm,
            gridTemplateColumns: 'minmax(0, 1fr)',
          },
          '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
          },
          '&:active': {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
          },
          '&:focus-visible': {
            outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
            outlineOffset: -2,
          },
        }}
      >
        <span css={{ display: 'flex', flex: 1, flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
          <span css={{ alignItems: 'baseline', display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm, minWidth: 0 }}>
            <Typography.Text bold css={{ minWidth: 0, overflowWrap: 'anywhere' }}>
              {answer.id}
            </Typography.Text>
            <Typography.Text size="sm" color="secondary" css={{ minWidth: 0, overflowWrap: 'anywhere' }}>
              <AnswerKind answer={answer} />
            </Typography.Text>
          </span>
        </span>
        <span
          css={{
            alignItems: 'center',
            display: 'flex',
            flexShrink: 0,
            gap: theme.spacing.md,
            minWidth: 0,
            '@container (max-width: 420px)': {
              justifyContent: 'space-between',
            },
          }}
        >
          <span
            css={{
              alignItems: 'flex-end',
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.xs,
              minWidth: 0,
              textAlign: 'right',
              width: '100%',
              '@container (max-width: 420px)': {
                alignItems: 'flex-start',
                textAlign: 'left',
              },
            }}
          >
            <AnswerSummary answer={answer} />
          </span>
          <span css={{ color: theme.colors.textSecondary, display: 'flex', flexShrink: 0 }} aria-hidden="true">
            {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          </span>
        </span>
      </button>
      {expanded && (
        <div
          id={detailsId}
          role="region"
          aria-labelledby={buttonId}
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            borderTop: `1px solid ${theme.colors.borderDecorative}`,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
            padding: theme.spacing.md,
          }}
        >
          <AnswerDetails answer={answer} />
        </div>
      )}
    </div>
  );
};

export const DecisionAnswersRenderer = ({
  answers,
}: {
  answers: DecisionAnswerViewModel[];
}): React.ReactElement | null => {
  const { theme } = useDesignSystemTheme();

  if (answers.length === 0) {
    return null;
  }

  return (
    <div
      role="list"
      data-testid="decision-answers"
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
      {answers.map((answer, index) => (
        <AnswerRow key={`${answer.id}-${index}`} answer={answer} />
      ))}
    </div>
  );
};
