import { describe, it, expect } from '@jest/globals';

import type { LabelSchema, LabelSchemaType } from '../../components/label-schemas';
import {
  buildPrefilledAnswers,
  buildPrefilledRationales,
  buildPriorAssessmentIds,
  extractPriorAnswers,
  type RawTraceAssessment,
} from './reviewAnswers';

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

  it('captures a rationale recorded alongside the value', () => {
    const raw: RawTraceAssessment[] = [
      { assessment_name: 'correctness', feedback: { value: true }, rationale: 'looks right' },
    ];
    expect(extractPriorAnswers(raw)).toEqual([
      { name: 'correctness', kind: 'feedback', value: true, rationale: 'looks right', valid: true },
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

  it('captures the assessment id', () => {
    const raw: RawTraceAssessment[] = [
      { assessment_id: 'a1', assessment_name: 'correctness', feedback: { value: true } },
    ];
    expect(extractPriorAnswers(raw)[0].assessmentId).toBe('a1');
  });

  describe('reviewer source scoping', () => {
    const raw: RawTraceAssessment[] = [
      {
        assessment_id: 'a1',
        assessment_name: 'correctness',
        feedback: { value: true },
        source: { source_id: 'alice', source_type: 'HUMAN' },
      },
      {
        assessment_id: 'a2',
        assessment_name: 'correctness',
        feedback: { value: false },
        source: { source_id: 'bob', source_type: 'HUMAN' },
      },
    ];

    it('keeps every answer when no reviewer source is given', () => {
      expect(extractPriorAnswers(raw).map((p) => p.assessmentId)).toEqual(['a1', 'a2']);
    });

    it("keeps only the reviewer's own answers when scoped (case-insensitive)", () => {
      expect(extractPriorAnswers(raw, 'ALICE')).toEqual([
        { name: 'correctness', kind: 'feedback', value: true, valid: true, assessmentId: 'a1' },
      ]);
    });

    it('drops answers without a matching source when scoped', () => {
      expect(extractPriorAnswers([{ assessment_name: 'x', feedback: { value: true } }], 'alice')).toEqual([]);
    });

    it('drops a non-HUMAN assessment that shares the reviewer source_id when scoped', () => {
      // An LLM judge or SDK-written feedback can collide on source_id (both
      // `default` on a no-auth server); only the reviewer's own HUMAN answer
      // should be adopted, never superseded as if it were theirs.
      const collide: RawTraceAssessment[] = [
        {
          assessment_id: 'judge',
          assessment_name: 'correctness',
          feedback: { value: true },
          source: { source_id: 'default', source_type: 'LLM_JUDGE' },
        },
        {
          assessment_id: 'mine',
          assessment_name: 'correctness',
          feedback: { value: false },
          source: { source_id: 'default', source_type: 'HUMAN' },
        },
      ];
      expect(extractPriorAnswers(collide, 'default').map((p) => p.assessmentId)).toEqual(['mine']);
    });

    it('prefills nothing for an empty source id rather than matching source-less answers', () => {
      // `''` can't identify a reviewer; guard against `sameUser('', undefined)` matching.
      expect(extractPriorAnswers(raw, '')).toEqual([]);
      expect(extractPriorAnswers([{ assessment_name: 'x', feedback: { value: true } }], '')).toEqual([]);
    });
  });
});

describe('buildPriorAssessmentIds', () => {
  const schemas = [schema('correctness', 'FEEDBACK')];

  it('maps each schema to the last matching valid prior assessment id', () => {
    const priors = extractPriorAnswers([
      { assessment_id: 'old', assessment_name: 'correctness', feedback: { value: true } },
      { assessment_id: 'new', assessment_name: 'correctness', feedback: { value: false } },
    ]);
    expect(buildPriorAssessmentIds(priors, schemas)).toEqual({ correctness: 'new' });
  });

  it('omits schemas with no prior answer', () => {
    expect(buildPriorAssessmentIds([], schemas)).toEqual({});
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

describe('buildPrefilledRationales', () => {
  const schemas = [schema('correctness', 'FEEDBACK')];

  it('prefills the rationale from the matching prior answer', () => {
    const priors = extractPriorAnswers([
      { assessment_name: 'correctness', feedback: { value: true }, rationale: 'looks right' },
    ]);
    expect(buildPrefilledRationales(priors, schemas)).toEqual({ correctness: 'looks right' });
  });

  it('omits schemas whose prior answer has no rationale', () => {
    const priors = extractPriorAnswers([{ assessment_name: 'correctness', feedback: { value: true } }]);
    expect(buildPrefilledRationales(priors, schemas)).toEqual({});
  });

  it('takes the most recent matching prior (last wins)', () => {
    const priors = extractPriorAnswers([
      { assessment_name: 'correctness', feedback: { value: true }, rationale: 'first' },
      { assessment_name: 'correctness', feedback: { value: false }, rationale: 'second' },
    ]);
    expect(buildPrefilledRationales(priors, schemas)).toEqual({ correctness: 'second' });
  });
});

describe('most-recent selection by timestamp (not array order)', () => {
  const schemas = [schema('correctness', 'FEEDBACK')];

  // The fresh assessment is FIRST in the array but carries the newer timestamp;
  // the stale one is LAST. The old `.at(-1)` array-order pick would wrongly
  // choose the stale one, since the response order is backend heap-dependent.
  const outOfOrder: RawTraceAssessment[] = [
    {
      assessment_id: 'fresh',
      assessment_name: 'correctness',
      feedback: { value: false },
      rationale: 'fresh',
      last_update_time: '2025-04-19T10:00:00.000Z',
    },
    {
      assessment_id: 'stale',
      assessment_name: 'correctness',
      feedback: { value: true },
      rationale: 'stale',
      last_update_time: '2025-04-19T08:00:00.000Z',
    },
  ];

  it('prefills the answer with the newest last_update_time, not the last in the array', () => {
    expect(buildPrefilledAnswers(extractPriorAnswers(outOfOrder), schemas)).toEqual({ correctness: false });
  });

  it('prefills the rationale from the newest assessment', () => {
    expect(buildPrefilledRationales(extractPriorAnswers(outOfOrder), schemas)).toEqual({ correctness: 'fresh' });
  });

  it('supersedes the newest assessment id', () => {
    expect(buildPriorAssessmentIds(extractPriorAnswers(outOfOrder), schemas)).toEqual({ correctness: 'fresh' });
  });

  it('falls back to create_time when last_update_time is absent', () => {
    const byCreateTime: RawTraceAssessment[] = [
      {
        assessment_id: 'fresh',
        assessment_name: 'correctness',
        feedback: { value: false },
        create_time: '2025-04-19T10:00:00.000Z',
      },
      {
        assessment_id: 'stale',
        assessment_name: 'correctness',
        feedback: { value: true },
        create_time: '2025-04-19T08:00:00.000Z',
      },
    ];
    expect(buildPriorAssessmentIds(extractPriorAnswers(byCreateTime), schemas)).toEqual({ correctness: 'fresh' });
  });

  it('falls back to array order when timestamps are equal', () => {
    // Equal timestamps -> last in array wins, preserving the old `.at(-1)` behavior.
    const sameTime: RawTraceAssessment[] = [
      {
        assessment_id: 'a1',
        assessment_name: 'correctness',
        feedback: { value: true },
        last_update_time: '2025-04-19T10:00:00.000Z',
      },
      {
        assessment_id: 'a2',
        assessment_name: 'correctness',
        feedback: { value: false },
        last_update_time: '2025-04-19T10:00:00.000Z',
      },
    ];
    expect(buildPriorAssessmentIds(extractPriorAnswers(sameTime), schemas)).toEqual({ correctness: 'a2' });
  });

  it('prefers a timestamped answer over an un-timestamped one regardless of array position', () => {
    // The timestamped (fresh) answer is FIRST; the un-timestamped one is LAST.
    // A missing timestamp sorts oldest, so the timestamped answer still wins.
    const mixed: RawTraceAssessment[] = [
      {
        assessment_id: 'timestamped',
        assessment_name: 'correctness',
        feedback: { value: false },
        last_update_time: '2025-04-19T10:00:00.000Z',
      },
      { assessment_id: 'untimed', assessment_name: 'correctness', feedback: { value: true } },
    ];
    expect(buildPriorAssessmentIds(extractPriorAnswers(mixed), schemas)).toEqual({ correctness: 'timestamped' });
  });

  it('treats an unparseable timestamp as missing (sorts oldest)', () => {
    const badTime: RawTraceAssessment[] = [
      {
        assessment_id: 'valid-time',
        assessment_name: 'correctness',
        feedback: { value: false },
        last_update_time: '2025-04-19T10:00:00.000Z',
      },
      {
        assessment_id: 'garbage-time',
        assessment_name: 'correctness',
        feedback: { value: true },
        last_update_time: 'not-a-date',
      },
    ];
    expect(buildPriorAssessmentIds(extractPriorAnswers(badTime), schemas)).toEqual({ correctness: 'valid-time' });
  });

  it('prefers last_update_time over create_time when both are present', () => {
    // The fresher last_update_time belongs to the assessment with the OLDER
    // create_time, so a create_time-based pick would choose the wrong one.
    const both: RawTraceAssessment[] = [
      {
        assessment_id: 'newer-update',
        assessment_name: 'correctness',
        feedback: { value: false },
        create_time: '2025-04-19T08:00:00.000Z',
        last_update_time: '2025-04-19T12:00:00.000Z',
      },
      {
        assessment_id: 'older-update',
        assessment_name: 'correctness',
        feedback: { value: true },
        create_time: '2025-04-19T10:00:00.000Z',
        last_update_time: '2025-04-19T11:00:00.000Z',
      },
    ];
    expect(buildPriorAssessmentIds(extractPriorAnswers(both), schemas)).toEqual({ correctness: 'newer-update' });
  });
});
