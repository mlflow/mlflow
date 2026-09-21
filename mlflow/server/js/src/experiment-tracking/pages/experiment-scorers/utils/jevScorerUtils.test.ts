import { describe, expect, it } from '@jest/globals';
import { parseJevCriteria } from './jevScorerUtils';
import {
  convertFormDataToScheduledScorer,
  transformScheduledScorer,
  transformScorerConfig,
} from './scorerTransformUtils';
import { getFormValuesFromScorer } from '../scorerCardUtils';
import type { JevScorer, JevAnswerType } from '../types';
import type { JevScorerFormData } from '../JevScorerFormRenderer';

describe('Jev scorer configuration', () => {
  it.each([
    ['noul', null, null],
    ['noul', { true: 'Correct', false: 'Incorrect' }, 0],
    ['choice', { billing: 'Payment questions', technical: 'Product questions' }, null],
    ['score', ['Incorrect', 'Partial', 'Correct'], null],
    ['score', ['  Multiline\ncriterion  ', ''], null],
  ])('round trips %s configuration through API and editable form', (answerType, criteria, threshold) => {
    const scorer: JevScorer = {
      name: 'quality',
      type: 'jev',
      model: 'gateway:/jev',
      question: 'Assess the response',
      answerType: answerType as JevAnswerType,
      criteria: criteria as JevScorer['criteria'],
      threshold: threshold as number | null,
      sampleRate: 25,
      filterString: "trace.status = 'OK'",
      version: 3,
      sdkConfig: { description: 'SDK description', timeout: 30, aggregations: ['mean'] },
    };
    const apiConfig = transformScheduledScorer(scorer);
    const loaded = transformScorerConfig({ ...apiConfig, scorer_version: 3 });
    const form = getFormValuesFromScorer(loaded) as JevScorerFormData;
    const edited = convertFormDataToScheduledScorer({ ...form, question: 'Updated question' }, loaded);
    const serialized = JSON.parse(transformScheduledScorer(edited).serialized_scorer);
    expect(edited).toMatchObject({ ...scorer, question: 'Updated question' });
    expect(serialized).toMatchObject(scorer.sdkConfig!);
    expect(serialized.jev_scorer_pydantic_data).toEqual({
      ...scorer.sdkConfig,
      name: 'quality',
      model: 'gateway:/jev',
      question: 'Updated question',
      answer_type: answerType,
      criteria,
      threshold,
    });
    expect(serialized.call_source).toBeUndefined();
    expect(serialized.instructions_judge_pydantic_data).toBeUndefined();
    expect(serialized.is_session_level_scorer).toBe(false);
  });

  it('loads a deleted endpoint as an empty selection without changing provider', () => {
    const loaded = transformScorerConfig({
      name: 'quality',
      serialized_scorer: JSON.stringify({
        jev_scorer_pydantic_data: { name: 'quality', model: null, question: 'Is it correct?', answer_type: 'noul' },
      }),
    });
    expect(loaded).toMatchObject({ type: 'jev', model: null });
    expect(getFormValuesFromScorer(loaded)).toMatchObject({ scorerType: 'jev', model: '' });
  });

  it('creates a native scorer without a base scorer', () => {
    const scorer = convertFormDataToScheduledScorer({
      scorerType: 'jev',
      name: 'relevance',
      model: 'gateway:/jev',
      question: 'Is the response relevant?',
      answerType: 'noul',
      criteria: '',
      threshold: '0.7',
      sampleRate: 0,
    });
    expect(scorer).toMatchObject({ type: 'jev', threshold: 0.7, criteria: null, sampleRate: 0 });
    expect(JSON.parse(transformScheduledScorer(scorer).serialized_scorer).jev_scorer_pydantic_data.threshold).toBe(0.7);
  });
});

describe('Jev criteria validation', () => {
  it.each([
    ['choice', '[]'],
    ['choice', '{}'],
    ['choice', '{"a": 2}'],
    ['choice', '{"": "empty"}'],
    ['noul', '{"yes": "Unsupported key"}'],
    ['score', '["One level"]'],
    ['score', JSON.stringify(Array(11).fill('level'))],
  ])('rejects invalid %s criteria %s', (answerType, criteria) => {
    expect(() => parseJevCriteria(answerType as JevAnswerType, criteria)).toThrow();
  });

  it('accepts a single choice and preserves label punctuation', () => {
    expect(parseJevCriteria('choice', '{"a:b=c": "Description"}')).toEqual({ 'a:b=c': 'Description' });
  });
});
