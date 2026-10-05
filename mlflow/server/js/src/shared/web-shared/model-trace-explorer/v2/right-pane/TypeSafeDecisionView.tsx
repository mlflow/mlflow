import { ChevronRightIcon, Progress, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { Assessment } from '../ModelTrace.types';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';

type Entry = { id: string; value: unknown };
export type TypeSafeDecision = { questions: Entry[] | null; answers: Entry[] };

type ScoreLevel = { score: number; description?: unknown; probability?: number };
type DistributionRow = {
  label: string;
  accessibleLabel: string;
  probability?: number;
  emphasized?: boolean;
  description?: unknown;
};
type Answer =
  | { kind: 'noul'; id: string; probabilityTrue: number }
  | { kind: 'choice'; id: string; choice: string; confidence: number; probabilities: [string, number][] }
  | { kind: 'score'; id: string; score: number; confidence: number; levels: ScoreLevel[] }
  | { kind: 'unknown'; id: string; rawAnswer: unknown; declaredType?: string };

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const isProbability = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;

export const resolveTypeSafeDecision = (
  span?: { chatMessageFormat?: unknown; inputs?: unknown; outputs?: unknown } | null,
): TypeSafeDecision | null => {
  if (span?.chatMessageFormat !== 'typesafe') return null;
  const answers = asRecord(asRecord(span.outputs)?.['answers']);
  if (!answers || Object.keys(answers).length === 0) return null;

  const questions = asRecord(asRecord(span.inputs)?.['questions']);
  return {
    questions: questions ? Object.entries(questions).map(([id, value]) => ({ id, value })) : null,
    answers: Object.entries(answers).map(([id, value]) => ({ id, value })),
  };
};

const asProbabilities = (value: unknown): [string, number][] | null => {
  const record = asRecord(value);
  if (!record) return null;
  const probabilities: [string, number][] = [];
  for (const [label, probability] of Object.entries(record)) {
    if (!isProbability(probability)) return null;
    probabilities.push([label, probability]);
  }
  return probabilities;
};

const getScoreLevels = (legendValue: unknown, probabilitiesValue: unknown): ScoreLevel[] | null => {
  const legend = asRecord(legendValue);
  const probabilities = asRecord(probabilitiesValue);
  if (!legend || !probabilities) return null;

  const levels: ScoreLevel[] = [];
  for (const key of new Set([...Object.keys(legend), ...Object.keys(probabilities)])) {
    const score = Number(key);
    const probability = probabilities[key];
    if (!Number.isSafeInteger(score) || score < 0 || (probability !== undefined && !isProbability(probability))) {
      return null;
    }
    levels.push({ score, description: legend[key], probability });
  }
  return levels.length ? levels.sort((left, right) => left.score - right.score) : null;
};

const parseAnswer = ({ id, value }: Entry): Answer => {
  const raw = asRecord(value);
  const type = raw?.['type'];
  if (raw && type === 'noul' && isProbability(raw['noul'])) {
    return { kind: 'noul', id, probabilityTrue: raw['noul'] };
  }
  if (raw && type === 'choice') {
    const choice = raw['choice'];
    const confidence = raw['confidence'];
    const probabilities = asProbabilities(raw['probabilities']);
    if (typeof choice === 'string' && isProbability(confidence) && probabilities) {
      return { kind: 'choice', id, choice, confidence, probabilities };
    }
  }
  if (raw && type === 'score') {
    const score = raw['score'];
    const confidence = raw['confidence'];
    const levels = getScoreLevels(raw['legend'], raw['probabilities']);
    if (typeof score === 'number' && Number.isFinite(score) && isProbability(confidence) && levels) {
      return { kind: 'score', id, score, confidence, levels };
    }
  }
  return {
    kind: 'unknown',
    id,
    rawAnswer: value,
    declaredType: typeof type === 'string' ? type : undefined,
  };
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

const QuestionRow = ({ entry }: { entry: Entry }) => {
  const question = asRecord(entry.value);
  const instructions = question?.['instructions'];
  const criteria = question?.['criteria'];
  const declaredType = typeof question?.['type'] === 'string' ? question['type'] : undefined;
  const hasStructuredInstructions = instructions !== undefined && typeof instructions !== 'string';
  const expandable = hasStructuredInstructions || criteria !== undefined;
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
          {hasStructuredInstructions && (
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
        </div>
      </details>
    </div>
  );
};

export const TypeSafeDecisionInputs = ({
  decision,
  fields,
  assessments,
}: {
  decision: TypeSafeDecision;
  fields: readonly { key: string; value: string }[];
  assessments?: Assessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const state = fields.find(({ key }) => key === 'state');
  const rawQuestions = fields.find(({ key }) => key === 'questions');
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {state && (
        <ModelTraceExplorerFieldRenderer
          title={state.key}
          data={state.value}
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
              <QuestionRow key={entry.id + '-' + index} entry={entry} />
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
  if (answer.kind === 'choice') {
    return <FormattedMessage defaultMessage="Choice" description="Choice decision answer type" />;
  }
  if (answer.kind === 'score') {
    return <FormattedMessage defaultMessage="Score" description="Score decision answer type" />;
  }
  if (answer.kind === 'noul') {
    return <FormattedMessage defaultMessage="Noul" description="Noul decision answer type" />;
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
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  if (answer.kind === 'unknown') {
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
  if (answer.kind === 'noul') {
    const predictsTrue = answer.probabilityTrue >= 0.5;
    rows = [
      {
        label: intl.formatMessage({ defaultMessage: 'True', description: 'True outcome in a decision Noul answer' }),
        accessibleLabel: probabilityFor('true'),
        probability: answer.probabilityTrue,
        emphasized: predictsTrue,
      },
      {
        label: intl.formatMessage({ defaultMessage: 'False', description: 'False outcome in a decision Noul answer' }),
        accessibleLabel: probabilityFor('false'),
        probability: 1 - answer.probabilityTrue,
        emphasized: !predictsTrue,
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
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Text size="sm" bold>
          <FormattedMessage
            defaultMessage="Probability distribution"
            description="Heading for a decision answer probability distribution"
          />
        </Typography.Text>
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          {rows.map(({ label, probability, description, accessibleLabel, emphasized }) => (
            <ProbabilityRow
              key={label}
              accessibleLabel={accessibleLabel}
              probability={probability}
              emphasized={emphasized}
              label={
                answer.kind === 'score' ? (
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
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
      {answer.kind !== 'noul' && <ConfidenceFooter value={answer.confidence} />}
    </>
  );
};

const AnswerRow = ({ entry }: { entry: Entry }) => {
  const { theme } = useDesignSystemTheme();
  const answer = parseAnswer(entry);
  return (
    <div role="listitem">
      <details>
        <summary
          css={{
            alignItems: 'center',
            backgroundColor: theme.colors.backgroundPrimary,
            color: theme.colors.textPrimary,
            cursor: 'pointer',
            display: 'grid',
            gap: theme.spacing.md,
            gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)',
            listStyle: 'none',
            padding: theme.spacing.md,
            '@container (max-width: 420px)': {
              alignItems: 'stretch',
              gap: theme.spacing.sm,
              gridTemplateColumns: 'minmax(0, 1fr)',
            },
            '&::-webkit-details-marker': { display: 'none' },
            '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
            '&:active': { backgroundColor: theme.colors.actionDefaultBackgroundPress },
            '&:focus-visible': {
              outline: '2px solid ' + theme.colors.actionDefaultBorderFocus,
              outlineOffset: -2,
            },
          }}
        >
          <span css={{ alignItems: 'baseline', display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm, minWidth: 0 }}>
            <Typography.Text bold css={{ minWidth: 0, overflowWrap: 'anywhere' }}>
              {answer.id}
            </Typography.Text>
            <Typography.Text size="sm" color="secondary" css={{ minWidth: 0, overflowWrap: 'anywhere' }}>
              <AnswerKind answer={answer} />
            </Typography.Text>
          </span>
          <span
            css={{
              alignItems: 'center',
              display: 'flex',
              gap: theme.spacing.md,
              minWidth: 0,
              '@container (max-width: 420px)': { justifyContent: 'space-between' },
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
                '@container (max-width: 420px)': { alignItems: 'flex-start', textAlign: 'left' },
              }}
            >
              <AnswerSummary answer={answer} />
            </span>
            <span
              css={{
                color: theme.colors.textSecondary,
                display: 'flex',
                flexShrink: 0,
                transition: 'transform 150ms ease',
                'details[open] &': { transform: 'rotate(90deg)' },
              }}
              aria-hidden="true"
            >
              <ChevronRightIcon />
            </span>
          </span>
        </summary>
        <div
          className="decision-answer-detail"
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            borderTop: '1px solid ' + theme.colors.borderDecorative,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
            padding: theme.spacing.md,
          }}
        >
          <AnswerDetails answer={answer} />
        </div>
      </details>
    </div>
  );
};

export const TypeSafeDecisionAnswers = ({ decision }: { decision: TypeSafeDecision }) => (
  <DecisionList testId="decision-answers">
    {decision.answers.map((entry, index) => (
      <AnswerRow key={entry.id + '-' + index} entry={entry} />
    ))}
  </DecisionList>
);
