export type Entry = { id: string; value: unknown };
export type Decision = { source: 'typesafe' | 'openai_decisions'; questions: Entry[] | null; answers: Entry[] };
export type DecisionSpan = { chatMessageFormat?: unknown; inputs?: unknown; outputs?: unknown };

export type ScoreLevel = { score: number; description?: unknown; probability?: number };
export type Answer =
  | { kind: 'noul'; id: string; probabilityTrue: number }
  | { kind: 'predicate'; id: string; probabilityTrue: number }
  | { kind: 'choice'; id: string; choice: string; confidence: number; probabilities: [string, number][] }
  | { kind: 'score'; id: string; score: number; confidence: number; levels: ScoreLevel[] }
  | { kind: 'refusal'; id: string; rawAnswer: unknown }
  | { kind: 'unknown'; id: string; rawAnswer: unknown; declaredType?: string };

export const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

export const isProbability = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;

export const hasUnitProbabilityTotal = (probabilities: number[]) =>
  Math.abs(probabilities.reduce((total, probability) => total + probability, 0) - 1) <= 0.01 + Number.EPSILON;
