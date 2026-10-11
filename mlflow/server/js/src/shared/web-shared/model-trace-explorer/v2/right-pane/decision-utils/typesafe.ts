import { asRecord, isProbability } from './shared';
import type { Answer, Decision, DecisionSpan, Entry, ScoreLevel } from './shared';

export const resolveTypeSafeDecision = (span?: DecisionSpan | null): Decision | null => {
  if (span?.chatMessageFormat !== 'typesafe') return null;
  const answers = asRecord(asRecord(span.outputs)?.['answers']);
  if (!answers || Object.keys(answers).length === 0) return null;

  const questions = asRecord(asRecord(span.inputs)?.['questions']);
  return {
    source: 'typesafe',
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

  const keys = Object.keys(legend);
  if (keys.length === 0 || keys.length !== Object.keys(probabilities).length) return null;

  const levels: ScoreLevel[] = [];
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(probabilities, key)) return null;
    const score = Number(key);
    const probability = probabilities[key];
    if (!Number.isSafeInteger(score) || score < 0 || !isProbability(probability)) {
      return null;
    }
    levels.push({ score, description: legend[key], probability });
  }
  return levels.sort((left, right) => left.score - right.score);
};

export const parseTypeSafeAnswer = ({ id, value }: Entry): Answer => {
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
