import { describe, it, expect } from '@jest/globals';

import type { LabelSchema, LabelSchemaType } from '../../components/label-schemas';
import { buildPrefilledAnswers, extractPriorAnswers, type RawTraceAssessment } from './reviewAnswers';

const schema = (name: string, type: LabelSchemaType): LabelSchema => ({
  schema_id: `ls-${name}`,
  experiment_id: '1',
  name,
  type,
  input: { pass_fail: { positive_label: 'Yes', negative_label: 'No' } },
});

describe('extractPriorAnswers', () => {
  it('normalizes feedback and expectation assessments', () => {
    const raw: RawTraceAssessment[] = [
      { assessment_name: 'correctness', feedback: { value: true } },
      { assessment_name: 'expected', expectation: { value: 'Paris' } },
    ];
    expect(extractPriorAnswers(raw)).toEqual([
      { name: 'correctness', kind: 'feedback', value: true, valid: true },
      { name: 'expected', kind: 'expectation', value: 'Paris', valid: true },
    ]);
  });

  it('reads a serialized expectation value', () => {
    const raw: RawTraceAssessment[] = [
      { assessment_name: 'expected', expectation: { serialized_value: { value: '42' } } },
    ];
    expect(extractPriorAnswers(raw)).toEqual([{ name: 'expected', kind: 'expectation', value: '42', valid: true }]);
  });

  it('marks valid:false assessments and drops nameless / valueless / non-answer ones', () => {
    const raw: RawTraceAssessment[] = [
      { assessment_name: 'tone', feedback: { value: 'terse' }, valid: false },
      { feedback: { value: true } }, // no name
      { assessment_name: 'blank', feedback: {} }, // no value
      { assessment_name: 'an-issue' }, // neither feedback nor expectation
    ];
    expect(extractPriorAnswers(raw)).toEqual([{ name: 'tone', kind: 'feedback', value: 'terse', valid: false }]);
  });

  it('keeps falsy-but-present values and drops empty ones', () => {
    const raw: RawTraceAssessment[] = [
      { assessment_name: 'flag', feedback: { value: false } },
      { assessment_name: 'score', feedback: { value: 0 } },
      { assessment_name: 'opts', feedback: { value: ['a', 'b'] } },
      { assessment_name: 'empty-str', feedback: { value: '' } },
      { assessment_name: 'empty-arr', feedback: { value: [] } },
      { assessment_name: 'null-val', expectation: { value: null } },
    ];
    expect(extractPriorAnswers(raw)).toEqual([
      { name: 'flag', kind: 'feedback', value: false, valid: true },
      { name: 'score', kind: 'feedback', value: 0, valid: true },
      { name: 'opts', kind: 'feedback', value: ['a', 'b'], valid: true },
    ]);
  });

  it('falls back to serialized_value when expectation.value is undefined', () => {
    const raw: RawTraceAssessment[] = [
      { assessment_name: 'expected', expectation: { value: undefined, serialized_value: { value: 'x' } } },
    ];
    expect(extractPriorAnswers(raw)).toEqual([{ name: 'expected', kind: 'expectation', value: 'x', valid: true }]);
  });
});

describe('buildPrefilledAnswers', () => {
  const schemas = [schema('correctness', 'FEEDBACK'), schema('expected', 'EXPECTATION')];

  it('matches a prior answer by name and kind', () => {
    const priors = extractPriorAnswers([
      { assessment_name: 'correctness', feedback: { value: false } },
      { assessment_name: 'expected', expectation: { value: 'Paris' } },
    ]);
    expect(buildPrefilledAnswers(priors, schemas)).toEqual({ correctness: false, expected: 'Paris' });
  });

  it('does not adopt a same-named prior of the wrong kind', () => {
    // An expectation named "correctness" must not prefill a FEEDBACK schema.
    const priors = extractPriorAnswers([{ assessment_name: 'correctness', expectation: { value: 'x' } }]);
    expect(buildPrefilledAnswers(priors, schemas)).toEqual({});
  });

  it('skips invalid priors', () => {
    const priors = extractPriorAnswers([{ assessment_name: 'correctness', feedback: { value: true }, valid: false }]);
    expect(buildPrefilledAnswers(priors, schemas)).toEqual({});
  });

  it('takes the most recent matching prior (last wins)', () => {
    const priors = extractPriorAnswers([
      { assessment_name: 'correctness', feedback: { value: true } },
      { assessment_name: 'correctness', feedback: { value: false } },
    ]);
    expect(buildPrefilledAnswers(priors, schemas)).toEqual({ correctness: false });
  });

  it('leaves unmatched schemas unanswered', () => {
    expect(buildPrefilledAnswers([], schemas)).toEqual({});
  });

  it('prefills array and numeric values intact', () => {
    const multiSchemas = [schema('opts', 'FEEDBACK'), schema('score', 'FEEDBACK')];
    const priors = extractPriorAnswers([
      { assessment_name: 'opts', feedback: { value: ['a', 'b'] } },
      { assessment_name: 'score', feedback: { value: 4 } },
    ]);
    expect(buildPrefilledAnswers(priors, multiSchemas)).toEqual({ opts: ['a', 'b'], score: 4 });
  });
});
