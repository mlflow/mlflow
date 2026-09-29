import type {
  DecisionAnswerViewModel,
  DecisionChoiceProbability,
  DecisionQuestionViewModel,
  DecisionScoreLevel,
  DecisionSpanLike,
  DecisionTranslator,
  DecisionUnknownAnswerViewModel,
  DecisionViewModel,
} from '../Decision.types';

type UnknownRecord = Record<string, unknown>;

const matches = ({ title, modelProvider }: DecisionSpanLike): boolean =>
  (typeof modelProvider === 'string' && modelProvider.toLowerCase() === 'typesafe') || title === 'typesafe.system_one';

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasOwn = (value: UnknownRecord, key: string): boolean => Object.prototype.hasOwnProperty.call(value, key);

const isProbability = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;

const parseQuestion = (id: string, rawQuestion: unknown): DecisionQuestionViewModel => {
  if (!isRecord(rawQuestion)) {
    return { id, rawQuestion };
  }

  const question: DecisionQuestionViewModel = { id, rawQuestion };
  if (typeof rawQuestion['type'] === 'string') {
    question.declaredType = rawQuestion['type'];
  }
  if (hasOwn(rawQuestion, 'instructions')) {
    question.instructions = rawQuestion['instructions'];
  }
  if (hasOwn(rawQuestion, 'criteria')) {
    question.criteria = rawQuestion['criteria'];
  }
  return question;
};

const parseQuestions = (value: unknown): DecisionQuestionViewModel[] =>
  isRecord(value) ? Object.entries(value).map(([id, rawQuestion]) => parseQuestion(id, rawQuestion)) : [];

const parseProbabilityMap = (value: unknown): DecisionChoiceProbability[] | null => {
  if (!isRecord(value)) {
    return null;
  }

  const probabilities: DecisionChoiceProbability[] = [];
  for (const [label, probability] of Object.entries(value)) {
    if (!isProbability(probability)) {
      return null;
    }
    probabilities.push({ label, probability });
  }
  return probabilities;
};

const parseScoreKey = (key: string): number | null => {
  if (!/^(0|[1-9]\d*)$/.test(key)) {
    return null;
  }
  const score = Number(key);
  return Number.isSafeInteger(score) ? score : null;
};

const parseScoreLevels = (
  legendValue: unknown,
  probabilitiesValue: unknown,
): { levels: DecisionScoreLevel[]; min: number; max: number } | null => {
  if (!isRecord(legendValue) || !isRecord(probabilitiesValue)) {
    return null;
  }

  const levelsByScore = new Map<number, DecisionScoreLevel>();
  for (const [key, description] of Object.entries(legendValue)) {
    const score = parseScoreKey(key);
    if (score === null) {
      return null;
    }
    levelsByScore.set(score, { score, description });
  }

  for (const [key, probability] of Object.entries(probabilitiesValue)) {
    const score = parseScoreKey(key);
    if (score === null || !isProbability(probability)) {
      return null;
    }
    const level = levelsByScore.get(score);
    if (level) {
      level.probability = probability;
    } else {
      levelsByScore.set(score, { score, probability });
    }
  }

  const levels = Array.from(levelsByScore.values()).sort((left, right) => left.score - right.score);
  if (levels.length === 0) {
    return null;
  }
  return { levels, min: levels[0].score, max: levels[levels.length - 1].score };
};

const unknownAnswer = (
  id: string,
  rawAnswer: unknown,
  declaredType: string | undefined,
  reason: 'malformed' | 'unsupported',
): DecisionUnknownAnswerViewModel => ({
  kind: 'unknown',
  id,
  rawAnswer,
  declaredType,
  reason,
});

const parseAnswer = (id: string, rawAnswer: unknown): DecisionAnswerViewModel => {
  if (!isRecord(rawAnswer) || typeof rawAnswer['type'] !== 'string') {
    return unknownAnswer(id, rawAnswer, undefined, 'malformed');
  }

  const declaredType = rawAnswer['type'];
  if (declaredType === 'noul') {
    return isProbability(rawAnswer['noul'])
      ? { kind: 'noul', id, rawAnswer, probabilityTrue: rawAnswer['noul'] }
      : unknownAnswer(id, rawAnswer, declaredType, 'malformed');
  }

  if (declaredType === 'choice') {
    const probabilities = parseProbabilityMap(rawAnswer['probabilities']);
    return typeof rawAnswer['choice'] === 'string' && isProbability(rawAnswer['confidence']) && probabilities !== null
      ? {
          kind: 'choice',
          id,
          rawAnswer,
          choice: rawAnswer['choice'],
          confidence: rawAnswer['confidence'],
          probabilities,
        }
      : unknownAnswer(id, rawAnswer, declaredType, 'malformed');
  }

  if (declaredType === 'score') {
    const parsedLevels = parseScoreLevels(rawAnswer['legend'], rawAnswer['probabilities']);
    const score = rawAnswer['score'];
    if (
      typeof score === 'number' &&
      Number.isFinite(score) &&
      isProbability(rawAnswer['confidence']) &&
      parsedLevels !== null &&
      score >= parsedLevels.min &&
      score <= parsedLevels.max
    ) {
      return {
        kind: 'score',
        id,
        rawAnswer,
        score,
        confidence: rawAnswer['confidence'],
        levels: parsedLevels.levels,
        range: { min: parsedLevels.min, max: parsedLevels.max },
      };
    }
    return unknownAnswer(id, rawAnswer, declaredType, 'malformed');
  }

  return unknownAnswer(id, rawAnswer, declaredType, 'unsupported');
};

const translate = ({ inputs, outputs }: DecisionSpanLike): DecisionViewModel | null => {
  if (!isRecord(outputs) || !isRecord(outputs['answers'])) {
    return null;
  }

  const answerEntries = Object.entries(outputs['answers']);
  if (answerEntries.length === 0) {
    return null;
  }

  const inputRecord = isRecord(inputs) ? inputs : null;
  const questions = parseQuestions(inputRecord?.['questions']);
  return {
    inputs: {
      fields: [
        { kind: 'field', fieldKey: 'state' },
        questions.length > 0
          ? { kind: 'questions', fieldKey: 'questions', items: questions }
          : { kind: 'field', fieldKey: 'questions' },
      ],
    },
    answers: answerEntries.map(([id, rawAnswer]) => parseAnswer(id, rawAnswer)),
  };
};

export const typeSafeSystemOneTranslator: DecisionTranslator = {
  matches,
  translate,
};
