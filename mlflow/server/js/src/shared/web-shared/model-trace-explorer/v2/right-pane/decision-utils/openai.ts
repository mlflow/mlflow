import { asRecord, hasUnitProbabilityTotal, isProbability } from './shared';
import type { Answer, Decision, DecisionSpan, Entry, ScoreLevel } from './shared';

const isChoiceValue = (value: unknown): value is string | boolean =>
  typeof value === 'string' || typeof value === 'boolean';

const choiceValueKey = (value: string | boolean) => `${typeof value}:${value}`;

export const parseOpenAIAnswer = ({ id, value }: Entry): Answer => {
  const raw = asRecord(value);
  const type = raw?.['type'];
  const name = raw?.['name'];
  const unknown: Answer = {
    kind: 'unknown',
    id,
    rawAnswer: value,
    declaredType: typeof type === 'string' ? type : undefined,
  };
  if (!raw || (name !== undefined && name !== null && typeof name !== 'string')) return unknown;

  if (type === 'predicate' && isProbability(raw['probability'])) {
    return { kind: 'predicate', id, probabilityTrue: raw['probability'] };
  }
  if (type === 'refusal') return { kind: 'refusal', id, rawAnswer: value };

  if (type === 'choice') {
    const choice = raw['choice'];
    const confidence = raw['confidence'];
    const probabilities = raw['probabilities'];
    if (!isChoiceValue(choice) || !isProbability(confidence) || !Array.isArray(probabilities)) return unknown;

    const rows: { value: string | boolean; probability: number }[] = [];
    const seen = new Set<string>();
    for (const item of probabilities) {
      const row = asRecord(item);
      const value = row?.['value'];
      const probability = row?.['probability'];
      if (!isChoiceValue(value) || !isProbability(probability)) return unknown;
      const key = choiceValueKey(value);
      if (seen.has(key)) return unknown;
      seen.add(key);
      rows.push({ value, probability });
    }
    if (!rows.length || !seen.has(choiceValueKey(choice))) return unknown;
    if (!hasUnitProbabilityTotal(rows.map((row) => row.probability))) return unknown;

    // A string "true" and the Boolean true are distinct choices in the API.
    const label = (value: string | boolean) =>
      typeof value === 'string' && rows.some((row) => typeof row.value === 'boolean' && String(row.value) === value)
        ? JSON.stringify(value)
        : String(value);
    return {
      kind: 'choice',
      id,
      choice: label(choice),
      confidence,
      probabilities: rows.map((row) => [label(row.value), row.probability]),
    };
  }

  if (type === 'score') {
    const score = raw['score'];
    const confidence = raw['confidence'];
    const probabilities = raw['probabilities'];
    if (typeof score !== 'number' || !Number.isFinite(score) || !isProbability(confidence)) return unknown;
    if (!Array.isArray(probabilities) || !probabilities.length) return unknown;

    const levels: ScoreLevel[] = [];
    const seen = new Set<number>();
    for (const item of probabilities) {
      const row = asRecord(item);
      const value = row?.['value'];
      const label = row?.['label'];
      const probability = row?.['probability'];
      if (typeof value !== 'number' || !Number.isSafeInteger(value) || typeof label !== 'string') return unknown;
      if (!isProbability(probability) || seen.has(value)) return unknown;
      seen.add(value);
      levels.push({ score: value, description: label, probability });
    }
    if (!hasUnitProbabilityTotal(levels.map((level) => level.probability ?? 0))) return unknown;
    return { kind: 'score', id, score, confidence, levels: levels.sort((left, right) => left.score - right.score) };
  }
  return unknown;
};

const isOpenAIQuestion = (value: unknown): boolean => {
  const question = asRecord(value);
  if (
    !question ||
    typeof question['instructions'] !== 'string' ||
    (question['name'] !== undefined && typeof question['name'] !== 'string')
  ) {
    return false;
  }
  if (question['type'] === 'predicate') return true;
  if (question['type'] === 'choice') {
    const choices = question['choices'];
    if (!Array.isArray(choices) || !choices.length) return false;
    const seen = new Set<string>();
    for (const item of choices) {
      const choice = asRecord(item);
      const value = choice?.['value'];
      if (
        !isChoiceValue(value) ||
        (choice?.['description'] !== undefined && typeof choice['description'] !== 'string')
      ) {
        return false;
      }
      const key = choiceValueKey(value);
      if (seen.has(key)) return false;
      seen.add(key);
    }
    return true;
  }
  if (question['type'] === 'score') {
    const levels = question['levels'];
    return (
      Array.isArray(levels) &&
      levels.length > 0 &&
      levels.every((item) => {
        const level = asRecord(item);
        return (
          level &&
          typeof level['label'] === 'string' &&
          (level['description'] === undefined || typeof level['description'] === 'string')
        );
      })
    );
  }
  return false;
};

const matchesOpenAIQuestion = (entry: Entry, questionValue: unknown): boolean => {
  const answer = parseOpenAIAnswer(entry);
  const question = asRecord(questionValue);
  const response = asRecord(entry.value);
  if (
    answer.kind === 'unknown' ||
    !question ||
    !response ||
    (question['name'] ?? null) !== (response['name'] ?? null)
  ) {
    return false;
  }
  if (answer.kind === 'refusal') return true;
  if (question['type'] !== answer.kind) return false;
  if (answer.kind === 'predicate') return true;

  if (answer.kind === 'choice') {
    const choices = question['choices'];
    const probabilities = response['probabilities'];
    if (!Array.isArray(choices) || !Array.isArray(probabilities) || choices.length !== probabilities.length) {
      return false;
    }
    const remaining = new Set<string>();
    for (const item of choices) {
      const value = asRecord(item)?.['value'];
      if (!isChoiceValue(value)) return false;
      remaining.add(choiceValueKey(value));
    }
    for (const item of probabilities) {
      const value = asRecord(item)?.['value'];
      if (!isChoiceValue(value) || !remaining.delete(choiceValueKey(value))) return false;
    }
    return remaining.size === 0;
  }

  if (answer.kind === 'score') {
    const levels = question['levels'];
    const probabilities = response['probabilities'];
    if (!Array.isArray(levels) || !Array.isArray(probabilities) || levels.length !== probabilities.length) {
      return false;
    }
    if (answer.score < 0 || answer.score > levels.length - 1) return false;
    const remaining = new Set(levels.map((_, index) => index));
    for (const item of probabilities) {
      const row = asRecord(item);
      const value = row?.['value'];
      if (typeof value !== 'number' || !Number.isSafeInteger(value) || !remaining.delete(value)) return false;
      if (row?.['label'] !== asRecord(levels[value])?.['label']) return false;
    }
    return remaining.size === 0;
  }
  return false;
};

export const resolveOpenAIDecision = (span?: DecisionSpan | null): Decision | null => {
  if (span?.chatMessageFormat !== 'openai_decisions') return null;
  const answers = asRecord(span.outputs)?.['answers'];
  if (!Array.isArray(answers) || answers.length === 0) return null;

  const inputQuestions = asRecord(span.inputs)?.['questions'];
  if (!Array.isArray(inputQuestions) || inputQuestions.length !== answers.length) return null;
  if (!inputQuestions.every(isOpenAIQuestion)) return null;
  const questions = inputQuestions.map((value, index) => ({
    id: (asRecord(value)?.['name'] as string) || `Question ${index + 1}`,
    value,
  }));
  const entries = answers.map((value, index) => ({
    id: (asRecord(value)?.['name'] as string) || questions[index].id,
    value,
  }));
  if (entries.some((entry, index) => !matchesOpenAIQuestion(entry, inputQuestions[index]))) return null;
  return { source: 'openai_decisions', questions, answers: entries };
};
