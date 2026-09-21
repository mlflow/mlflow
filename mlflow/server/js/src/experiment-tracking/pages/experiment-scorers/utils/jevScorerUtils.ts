import type { JevAnswerType, JevCriteria } from '../types';

export type JevCriteriaError = 'json' | 'score' | 'object' | 'descriptions' | 'noul' | 'choice';

class JevCriteriaValidationError extends Error {
  readonly code: JevCriteriaError;

  constructor(code: JevCriteriaError) {
    super(code);
    this.code = code;
  }
}

export function parseJevCriteria(answerType: JevAnswerType, text: string): JevCriteria {
  if (!text.trim() && answerType === 'noul') return null;
  let criteria: unknown;
  try {
    criteria = JSON.parse(text);
  } catch {
    throw new JevCriteriaValidationError('json');
  }
  if (answerType === 'score') {
    if (
      !Array.isArray(criteria) ||
      criteria.length < 2 ||
      criteria.length > 10 ||
      criteria.some((level) => typeof level !== 'string')
    ) {
      throw new JevCriteriaValidationError('score');
    }
    return criteria;
  }
  if (!criteria || typeof criteria !== 'object' || Array.isArray(criteria)) {
    throw new JevCriteriaValidationError('object');
  }
  const entries = Object.entries(criteria);
  if (entries.some(([label, description]) => !label.trim() || typeof description !== 'string')) {
    throw new JevCriteriaValidationError('descriptions');
  }
  if (answerType === 'noul' && entries.some(([label]) => label !== 'true' && label !== 'false')) {
    throw new JevCriteriaValidationError('noul');
  }
  if (answerType === 'choice' && (entries.length < 1 || entries.length > 255)) {
    throw new JevCriteriaValidationError('choice');
  }
  return criteria as Record<string, string>;
}

export function validateJevCriteria(answerType: JevAnswerType, text: string): true | JevCriteriaError {
  try {
    parseJevCriteria(answerType, text);
    return true;
  } catch (error) {
    if (error instanceof JevCriteriaValidationError) return error.code;
    throw error;
  }
}
