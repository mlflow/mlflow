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
});
