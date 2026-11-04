import { ChevronRightIcon, Progress, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { Assessment } from '../ModelTrace.types';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';

import { parseAnswer } from './decision-utils';
import { asRecord } from './decision-utils/shared';
import type { Answer, Decision, Entry } from './decision-utils/shared';

type DistributionRow = {
  label: string;
  accessibleLabel: string;
  probability?: number;
  emphasized?: boolean;
  description?: unknown;
};
const displayValue = (value: unknown): string => {
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2) ?? String(value);
};

const SerializedValue = ({ value }: { value: unknown }) => {
  if (typeof value !== 'object' || value === null) {
    return (
      <Typography.Text css={{ overflowWrap: 'anywhere', whiteSpace: 'pre-wrap' }}>
        {displayValue(value)}
      </Typography.Text>
    );
  }
  return <pre className="decision-json">{displayValue(value)}</pre>;
};

const DetailField = ({ label, value }: { label: React.ReactNode; value: unknown }) => (
  <div className="decision-detail-field">
    <Typography.Text bold size="sm">
      {label}
    </Typography.Text>
    <SerializedValue value={value} />
  </div>
);

const DecisionList = ({ children, testId }: { children: React.ReactNode; testId: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="list"
      data-testid={testId}
      css={{
        border: '1px solid ' + theme.colors.borderDecorative,
        borderRadius: theme.borders.borderRadiusMd,
        containerType: 'inline-size',
        overflow: 'hidden',
        '& > [role="listitem"] + [role="listitem"]': {
          borderTop: '1px solid ' + theme.colors.borderDecorative,
        },
        '& summary': { cursor: 'pointer', listStyle: 'none' },
        '& summary::-webkit-details-marker': { display: 'none' },
        '& summary:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
        '& summary:active': { backgroundColor: theme.colors.actionDefaultBackgroundPress },
        '& summary:focus-visible': {
          outline: '2px solid ' + theme.colors.actionDefaultBorderFocus,
          outlineOffset: -2,
        },
        '& .decision-chevron': {
          color: theme.colors.textSecondary,
          display: 'flex',
          flexShrink: 0,
          transition: 'transform 150ms ease',
        },
        '& details[open] .decision-chevron': { transform: 'rotate(90deg)' },
        '& .decision-detail': {
          backgroundColor: theme.colors.backgroundSecondary,
          borderTop: '1px solid ' + theme.colors.borderDecorative,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.md,
          padding: theme.spacing.md,
        },
        '& .decision-detail-field': {
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          minWidth: 0,
        },
        '& .decision-json': {
          backgroundColor: theme.colors.backgroundSecondary,
          border: '1px solid ' + theme.colors.borderDecorative,
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
        },
        '& .decision-answer-detail .decision-json': { backgroundColor: theme.colors.backgroundPrimary },
        '& .decision-question-header': {
          alignItems: 'center',
          display: 'flex',
          gap: theme.spacing.md,
          justifyContent: 'space-between',
          minWidth: 0,
          padding: theme.spacing.md,
        },
        '& .decision-question-main': {
          display: 'flex',
          flex: 1,
          flexDirection: 'column',
          gap: theme.spacing.xs,
          minWidth: 0,
        },
        '& .decision-question-title': {
          alignItems: 'baseline',
          display: 'flex',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
        },
        '& .decision-question-type': {
          backgroundColor: theme.colors.backgroundSecondary,
          borderRadius: theme.borders.borderRadiusSm,
          flexShrink: 0,
          padding: '0 ' + theme.spacing.xs + 'px',
        },
        '& .decision-question-preview': {
          display: '-webkit-box',
          overflow: 'hidden',
          overflowWrap: 'anywhere',
          whiteSpace: 'pre-wrap',
          WebkitBoxOrient: 'vertical',
          WebkitLineClamp: 2,
        },
        '& .decision-answer-header': {
          alignItems: 'center',
          backgroundColor: theme.colors.backgroundPrimary,
          color: theme.colors.textPrimary,
          display: 'grid',
          gap: theme.spacing.md,
          gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)',
          padding: theme.spacing.md,
        },
        '& .decision-answer-name': {
          alignItems: 'baseline',
          display: 'flex',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
          minWidth: 0,
          '& > *': { minWidth: 0, overflowWrap: 'anywhere' },
        },
        '& .decision-answer-side': {
          alignItems: 'center',
          display: 'flex',
          gap: theme.spacing.md,
          minWidth: 0,
        },
        '& .decision-answer-value': {
          alignItems: 'flex-end',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          minWidth: 0,
          textAlign: 'right',
          width: '100%',
        },
        '& .decision-probability-section': { display: 'flex', flexDirection: 'column', gap: theme.spacing.xs },
        '& .decision-probability-list': { display: 'flex', flexDirection: 'column' },
        '& .decision-probability-row': {
          alignItems: 'center',
          display: 'grid',
          gap: theme.spacing.sm,
          gridTemplateColumns: 'minmax(0, 1fr) minmax(64px, 2fr) min-content',
          minHeight: theme.general.heightSm,
        },
        '& .decision-probability-label': { minWidth: 0, overflowWrap: 'anywhere' },
        '& .decision-probability-bar': { minWidth: 0 },
        '& .decision-no-probability': { gridColumn: '2 / -1' },
        '& .decision-confidence': {
          alignItems: 'baseline',
          borderTop: '1px solid ' + theme.colors.borderDecorative,
          display: 'flex',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
          justifyContent: 'flex-end',
          paddingTop: theme.spacing.sm,
        },
        '& .decision-score-label': {
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          minWidth: 0,
        },
        '@container (max-width: 420px)': {
          '& .decision-answer-header': {
            alignItems: 'stretch',
            gap: theme.spacing.sm,
            gridTemplateColumns: 'minmax(0, 1fr)',
          },
          '& .decision-answer-side': { justifyContent: 'space-between' },
          '& .decision-answer-value': { alignItems: 'flex-start', textAlign: 'left' },
          '& .decision-probability-row': {
            alignItems: 'baseline',
            gridTemplateColumns: 'minmax(0, 1fr) min-content',
          },
          '& .decision-probability-bar': { gridColumn: '1 / -1', gridRow: 2 },
          '& .decision-no-probability': { gridColumn: 2, gridRow: 1 },
          '& .decision-percent': { gridColumn: 2, gridRow: 1 },
        },
      }}
    >
      {children}
    </div>
  );
};

const QuestionRow = ({ entry, source }: { entry: Entry; source: Decision['source'] }) => {
  const question = asRecord(entry.value);
  const instructions = question?.['instructions'];
  const criteria = question?.['criteria'];
  const choices = question?.['choices'];
  const levels = question?.['levels'];
  const declaredType = typeof question?.['type'] === 'string' ? question['type'] : undefined;
  const hasStructuredInstructions = instructions !== undefined && typeof instructions !== 'string';
  const showInstructionsInDetails =
    hasStructuredInstructions || (source === 'openai_decisions' && instructions !== undefined);
  const expandable =
    showInstructionsInDetails || criteria !== undefined || choices !== undefined || levels !== undefined;
  const instructionPreview = typeof instructions === 'string' && instructions.trim() ? instructions : undefined;

  const summary = (
    <span className="decision-question-header">
      <span className="decision-question-main">
        <span className="decision-question-title">
          <Typography.Text bold css={{ overflowWrap: 'anywhere' }}>
            {entry.id}
          </Typography.Text>
          <Typography.Text color="secondary" size="sm" className="decision-question-type">
            {declaredType ?? (
              <FormattedMessage defaultMessage="Unknown type" description="Unknown decision question type" />
            )}
          </Typography.Text>
        </span>
        {instructionPreview && (
          <Typography.Text color="secondary" size="sm" className="decision-question-preview">
            {instructionPreview}
          </Typography.Text>
        )}
      </span>
      {expandable && (
        <span className="decision-chevron" aria-hidden="true">
          <ChevronRightIcon />
        </span>
      )}
    </span>
  );

  if (!expandable) return <div role="listitem">{summary}</div>;
  return (
    <div role="listitem">
      <details>
        <summary>{summary}</summary>
        <div className="decision-detail">
          {showInstructionsInDetails && (
            <DetailField
              label={
                <FormattedMessage
                  defaultMessage="Instructions"
                  description="Label for decision question instructions"
                />
              }
              value={instructions}
            />
          )}
          {criteria !== undefined && (
            <DetailField
              label={<FormattedMessage defaultMessage="Criteria" description="Label for decision question criteria" />}
              value={criteria}
            />
          )}
          {choices !== undefined && (
            <DetailField
              label={<FormattedMessage defaultMessage="Choices" description="Label for decision question choices" />}
              value={choices}
            />
          )}
          {levels !== undefined && (
            <DetailField
              label={<FormattedMessage defaultMessage="Levels" description="Label for decision score levels" />}
              value={levels}
            />
          )}
        </div>
      </details>
    </div>
  );
};

export const DecisionInputs = ({
  decision,
  fields,
  assessments,
}: {
  decision: Decision;
  fields: readonly { key: string; value: string }[];
  assessments?: Assessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const evidence = fields.find(({ key }) => key === (decision.source === 'typesafe' ? 'state' : 'input'));
  const rawQuestions = fields.find(({ key }) => key === 'questions');
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {evidence && (
        <ModelTraceExplorerFieldRenderer
          title={evidence.key}
          data={evidence.value}
          renderMode="default"
          assessments={assessments}
        />
      )}
      {decision.questions?.length ? (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold color="secondary" size="sm">
            questions
          </Typography.Text>
          <DecisionList testId="decision-questions">
            {decision.questions.map((entry, index) => (
              <QuestionRow key={entry.id + '-' + index} entry={entry} source={decision.source} />
            ))}
          </DecisionList>
        </div>
      ) : (
        rawQuestions && (
          <ModelTraceExplorerFieldRenderer
            title={rawQuestions.key}
            data={rawQuestions.value}
            renderMode="default"
            assessments={assessments}
          />
        )
      )}
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
  const intl = useIntl();
  const formatted =
    probability === undefined
      ? undefined
      : intl.formatNumber(probability, { style: 'percent', maximumFractionDigits: 1 });
  return (
    <div className="decision-probability-row">
      <div className="decision-probability-label">{label}</div>
      <div
        className={
          probability === undefined ? 'decision-probability-bar decision-no-probability' : 'decision-probability-bar'
        }
      >
        {probability === undefined ? (
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="Not reported"
              description="Missing probability in a decision answer distribution"
            />
          </Typography.Text>
        ) : (
          <Progress.Root value={probability * 100} aria-label={accessibleLabel} aria-valuetext={formatted}>
            <Progress.Indicator />
          </Progress.Root>
        )}
      </div>
      {formatted && (
        <Typography.Text className="decision-percent" size="sm" color="secondary" bold={emphasized}>
          {formatted}
        </Typography.Text>
      )}
    </div>
  );
};

const ConfidenceFooter = ({ value }: { value: number }) => {
  const intl = useIntl();
  return (
    <div className="decision-confidence">
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

const AnswerKind = ({ answer }: { answer: Answer }) => {
  if (answer.kind === 'predicate') {
    return <FormattedMessage defaultMessage="Predicate" description="Predicate decision answer type" />;
  }
  if (answer.kind === 'choice') {
    return <FormattedMessage defaultMessage="Choice" description="Choice decision answer type" />;
  }
  if (answer.kind === 'score') {
    return <FormattedMessage defaultMessage="Score" description="Score decision answer type" />;
  }
  if (answer.kind === 'noul') {
    return <FormattedMessage defaultMessage="Noul" description="Noul decision answer type" />;
  }
  if (answer.kind === 'refusal') {
    return <FormattedMessage defaultMessage="Refusal" description="Refusal decision answer type" />;
  }
  return (
    <>
      {answer.declaredType ?? <FormattedMessage defaultMessage="Unknown" description="Unknown decision answer type" />}
    </>
  );
};

const AnswerSummary = ({ answer }: { answer: Answer }) => {
  const intl = useIntl();
  const percent = (value: number) => intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 1 });
  if (answer.kind === 'predicate') {
    return (
      <>
        <Typography.Text bold>{percent(answer.probabilityTrue)}</Typography.Text>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage defaultMessage="probability of true" description="Predicate decision probability summary" />
        </Typography.Text>
      </>
    );
  }
  if (answer.kind === 'refusal') {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Refused" description="Summary for a declined decision question" />
      </Typography.Text>
    );
  }
  if (answer.kind === 'unknown') {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Raw answer" description="Summary for an unrecognized decision answer" />
      </Typography.Text>
    );
  }
  const predictsTrue = answer.kind === 'noul' && answer.probabilityTrue >= 0.5;
  const value =
    answer.kind === 'noul'
      ? predictsTrue
        ? intl.formatMessage({
            defaultMessage: 'true',
            description: 'Lowercase true outcome in a decision Noul answer summary',
          })
        : intl.formatMessage({
            defaultMessage: 'false',
            description: 'Lowercase false outcome in a decision Noul answer summary',
          })
      : answer.kind === 'choice'
        ? answer.choice
        : intl.formatNumber(answer.score, { maximumFractionDigits: 2 });
  const metric =
    answer.kind === 'noul'
      ? intl.formatMessage(
          {
            defaultMessage: '{probability} probability',
            description: 'Compact probability summary for a decision Noul answer',
          },
          { probability: percent(predictsTrue ? answer.probabilityTrue : 1 - answer.probabilityTrue) },
        )
      : intl.formatMessage(
          {
            defaultMessage: '{confidence} confidence',
            description: 'Compact confidence summary for a decision answer',
          },
          { confidence: percent(answer.confidence) },
        );
  const range =
    answer.kind === 'score'
      ? intl.formatMessage(
          { defaultMessage: 'Range {min}–{max}', description: 'Score range summary for a decision answer' },
          {
            min: intl.formatNumber(answer.levels[0]?.score),
            max: intl.formatNumber(answer.levels[answer.levels.length - 1]?.score),
          },
        )
      : null;
  return (
    <>
      <Typography.Text bold size={answer.kind === 'score' ? 'lg' : undefined} css={{ overflowWrap: 'anywhere' }}>
        {value}
      </Typography.Text>
      {range && (
        <Typography.Text size="sm" color="secondary">
          {range}
        </Typography.Text>
      )}
      <Typography.Text size="sm" color="secondary">
        {metric}
      </Typography.Text>
    </>
  );
};

const AnswerDetails = ({ answer }: { answer: Answer }) => {
  const intl = useIntl();
  if (answer.kind === 'unknown' || answer.kind === 'refusal') {
    return (
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
  }

  const probabilityFor = (label: string) =>
    intl.formatMessage(
      { defaultMessage: 'Probability for {label}', description: 'Accessible label for a decision answer probability' },
      { label },
    );
  let rows: DistributionRow[];
  if (answer.kind === 'noul' || answer.kind === 'predicate') {
    const predictsTrue = answer.probabilityTrue >= 0.5;
    rows = [
      {
        label: intl.formatMessage({ defaultMessage: 'True', description: 'True outcome in a decision Noul answer' }),
        accessibleLabel: probabilityFor('true'),
        probability: answer.probabilityTrue,
        emphasized: answer.kind === 'noul' && predictsTrue,
      },
      {
        label: intl.formatMessage({ defaultMessage: 'False', description: 'False outcome in a decision Noul answer' }),
        accessibleLabel: probabilityFor('false'),
        probability: 1 - answer.probabilityTrue,
        emphasized: answer.kind === 'noul' && !predictsTrue,
      },
    ];
  } else if (answer.kind === 'choice') {
    rows = answer.probabilities.map(([label, probability]) => ({
      label,
      accessibleLabel: probabilityFor(label),
      probability,
      emphasized: label === answer.choice,
    }));
  } else {
    rows = answer.levels.map(({ score, description, probability }) => {
      const label = intl.formatNumber(score);
      return {
        label,
        description,
        probability,
        accessibleLabel: intl.formatMessage(
          {
            defaultMessage: 'Probability for score {score}',
            description: 'Accessible label for a decision score probability',
          },
          { score: label },
        ),
      };
    });
  }

  return (
    <>
      <div className="decision-probability-section">
        <Typography.Text size="sm" bold>
          <FormattedMessage
            defaultMessage="Probability distribution"
            description="Heading for a decision answer probability distribution"
          />
        </Typography.Text>
        <div className="decision-probability-list">
          {rows.map(({ label, probability, description, accessibleLabel, emphasized }) => (
            <ProbabilityRow
              key={label}
              accessibleLabel={accessibleLabel}
              probability={probability}
              emphasized={emphasized}
              label={
                answer.kind === 'score' ? (
                  <div className="decision-score-label">
                    <Typography.Text size="sm" bold>
                      {label}
                    </Typography.Text>
                    {description !== undefined && typeof description !== 'object' && (
                      <Typography.Text size="sm" color="secondary">
                        {displayValue(description)}
                      </Typography.Text>
                    )}
                    {description !== undefined && typeof description === 'object' && (
                      <SerializedValue value={description} />
                    )}
                  </div>
                ) : (
                  <Typography.Text size="sm" bold={emphasized} css={{ overflowWrap: 'anywhere' }}>
                    {label}
                  </Typography.Text>
                )
              }
            />
          ))}
        </div>
      </div>
      {(answer.kind === 'choice' || answer.kind === 'score') && <ConfidenceFooter value={answer.confidence} />}
    </>
  );
};

const AnswerRow = ({ entry, source }: { entry: Entry; source: Decision['source'] }) => {
  const answer = parseAnswer(entry, source);
  return (
    <div role="listitem">
      <details>
        <summary className="decision-answer-header">
          <span className="decision-answer-name">
            <Typography.Text bold>{answer.id}</Typography.Text>
            <Typography.Text size="sm" color="secondary">
              <AnswerKind answer={answer} />
            </Typography.Text>
          </span>
          <span className="decision-answer-side">
            <span className="decision-answer-value">
              <AnswerSummary answer={answer} />
            </span>
            <span className="decision-chevron" aria-hidden="true">
              <ChevronRightIcon />
            </span>
          </span>
        </summary>
        <div className="decision-answer-detail decision-detail">
          <AnswerDetails answer={answer} />
        </div>
      </details>
    </div>
  );
};

export const DecisionAnswers = ({ decision }: { decision: Decision }) => (
  <DecisionList testId="decision-answers">
    {decision.answers.map((entry, index) => (
      <AnswerRow key={entry.id + '-' + index} entry={entry} source={decision.source} />
    ))}
  </DecisionList>
);
