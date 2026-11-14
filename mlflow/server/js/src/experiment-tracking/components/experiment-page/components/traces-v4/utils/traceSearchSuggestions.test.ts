import { compileFilterModel } from './filterModel';
import { describe, expect, test } from '@jest/globals';
import type { ExpectationAssessment } from '@databricks/web-shared/model-trace-explorer';
import { FilterOp, type FilterFieldDef } from '@databricks/web-shared/traces-table';
import { makeFeedbackAssessment, makeIssueAssessment, makeTrace } from '../test-utils/mockTraces';
import { buildTraceSearchFields, getTraceSearchExamples, isSafeTraceSearchKey } from './traceSearchSuggestions';

const makeExpectation = (name: string, value: string | number | boolean): ExpectationAssessment => ({
  assessment_id: `expectation-${name}-${value}`,
  assessment_name: name,
  trace_id: 'trace-1',
  source: { source_type: 'HUMAN', source_id: 'test-user' },
  create_time: '2025-01-01T00:00:00Z',
  last_update_time: '2025-01-01T00:00:00Z',
  expectation: { value },
});

const textField = (id: string, operators: FilterOp[] = [FilterOp.EQUALS]): FilterFieldDef => ({
  id,
  label: id,
  operators,
  valueInput: 'text',
});

const keyedField = (
  id: 'tag' | 'metadata' | 'assessment' | 'expectation' | 'rationale' | 'span_attribute',
): FilterFieldDef => ({
  ...textField(id, [FilterOp.EQUALS, FilterOp.NOT_EQUALS]),
  requiresKey: true,
});

describe('buildTraceSearchFields', () => {
  test('offers rationale search for a scorer with rationale text', () => {
    const rationaleField = keyedField('rationale');
    rationaleField.operators = [FilterOp.CONTAINS];
    const fields = buildTraceSearchFields(
      [rationaleField],
      [
        makeTrace('trace-1', {
          assessments: [makeFeedbackAssessment('groundedness', 'no', { rationale: 'Unsupported claim' })],
        }),
      ],
    );
    expect(fields).toEqual([
      expect.objectContaining({
        label: 'rationale.groundedness',
        values: [{ value: 'Unsupported claim', label: 'Unsupported claim' }],
      }),
    ]);
  });

  test('keeps the span attribute namespace without using trace metadata as key suggestions', () => {
    const fields = buildTraceSearchFields(
      [keyedField('span_attribute')],
      [makeTrace('trace-a', { trace_metadata: { 'sidekick.skills.name': 'web.search' } })],
    );

    expect(fields).toHaveLength(1);
    expect(fields[0]).toMatchObject({
      id: 'span_attribute',
      label: 'attributes',
      values: [],
      aliases: ['attributes'],
    });
    expect(fields[0].key).toBeUndefined();
  });

  test.each([
    { key: 'sidekick.skills.name', expected: true },
    { key: 'bad"key', expected: false },
    { key: 'bad\\key', expected: false },
  ])('validates span attribute search key $key', ({ key, expected }) => {
    expect(isSafeTraceSearchKey('span_attribute', key)).toBe(expected);
  });

  test('collects real scalar, keyed, duration, and feedback values', () => {
    const traces = [
      makeTrace('trace-a', {
        execution_duration: '0s',
        state: 'OK',
        tags: { 'mlflow.traceName': 'chat', env: 'prod' },
        trace_metadata: {
          'mlflow.trace.session': 'session-a',
          'mlflow.trace.user': 'alice',
          'mlflow.source.name': 'app.py',
        },
        assessments: [
          makeFeedbackAssessment('quality', false),
          makeFeedbackAssessment('score', 0),
          {
            assessment_id: 'expected-1',
            assessment_name: 'expected',
            trace_id: 'trace-a',
            source: { source_type: 'HUMAN', source_id: 'test-user' },
            create_time: '2025-01-01T00:00:00Z',
            last_update_time: '2025-01-01T00:00:00Z',
            expectation: { value: 'yes' },
          } satisfies ExpectationAssessment,
          makeFeedbackAssessment('invalid', 'skip', { valid: false }),
          makeFeedbackAssessment('error', 'skip', {
            feedback: { error: { error_code: 'JUDGE_ERROR' } },
          }),
          makeIssueAssessment('iss-1'),
        ],
      }),
      makeTrace('trace-b', {
        execution_duration: '1.2s',
        state: 'ERROR',
        tags: { 'mlflow.traceName': 'completion', env: 'staging' },
        trace_metadata: {
          'mlflow.trace.session': 'session-b',
          'mlflow.trace.user': 'bob',
          'mlflow.source.name': 'worker.py',
        },
      }),
    ];
    const fields = buildTraceSearchFields(
      [
        {
          ...textField('duration', [FilterOp.EQUALS, FilterOp.GREATER_THAN_OR_EQUALS]),
          valueInput: 'number',
        },
        textField('state'),
        textField('trace_name'),
        textField('service_name'),
        textField('source'),
        textField('user'),
        textField('session'),
        keyedField('tag'),
        keyedField('assessment'),
      ],
      traces,
    );

    expect(fields.find((field) => field.id === 'duration')?.values).toEqual([
      { value: '0', label: '0 ms' },
      { value: '1200', label: '1200 ms' },
    ]);
    expect(fields.find((field) => field.id === 'state')?.values).toEqual([
      { value: 'ERROR', label: 'ERROR' },
      { value: 'OK', label: 'OK' },
    ]);
    expect(fields.find((field) => field.field.id === 'tag' && field.key === 'env')?.values).toEqual([
      { value: 'prod', label: 'prod' },
      { value: 'staging', label: 'staging' },
    ]);
    expect(fields.find((field) => field.field.id === 'assessment' && field.key === 'quality')?.values).toEqual([
      { value: 'false', label: 'false' },
    ]);
    // Numeric values aren't suggested (see below), but the name still is.
    expect(fields.find((field) => field.field.id === 'assessment' && field.key === 'score')?.values).toEqual([]);
    expect(
      fields.some(
        (field) =>
          field.key === 'expected' || field.key === 'invalid' || field.key === 'error' || field.key === 'issue-iss-1',
      ),
    ).toBe(false);
  });

  test('floors fractional milliseconds and keeps only range duration operators', () => {
    const fields = buildTraceSearchFields(
      [
        {
          ...textField('duration', [
            FilterOp.EQUALS,
            FilterOp.NOT_EQUALS,
            FilterOp.GREATER_THAN,
            FilterOp.LESS_THAN,
            FilterOp.GREATER_THAN_OR_EQUALS,
            FilterOp.LESS_THAN_OR_EQUALS,
          ]),
          valueInput: 'number',
        },
      ],
      [makeTrace('trace', { execution_duration: '1.234567891s' })],
    );

    expect(fields[0].field.operators).toEqual([
      FilterOp.GREATER_THAN,
      FilterOp.LESS_THAN,
      FilterOp.GREATER_THAN_OR_EQUALS,
      FilterOp.LESS_THAN_OR_EQUALS,
    ]);
    expect(fields[0].values).toEqual([{ value: '1234', label: '1234 ms' }]);
  });

  test('keeps scalar fields with empty values and emits no keyed field without data', () => {
    const duration = {
      ...textField('duration'),
      valueInput: 'number' as const,
    };
    const fields = buildTraceSearchFields([duration, keyedField('tag'), keyedField('assessment')], []);

    expect(fields).toHaveLength(1);
    expect(fields[0]).toMatchObject({
      id: 'duration',
      values: [],
      aliases: ['duration', 'latency'],
    });
    expect(getTraceSearchExamples(fields)).toEqual([]);
  });

  test('retains only keys the existing compilers can serialize', () => {
    const trace = makeTrace('trace', {
      tags: {
        'good.key': 'dot',
        'good key': 'space',
        'bad`key': 'unsafe',
        'mlflow.internal': 'internal',
      },
      trace_metadata: { owner: 'alice', 'bad key': 'unsafe' },
      assessments: [makeFeedbackAssessment('judge.score', 'good'), makeFeedbackAssessment('bad`name', 'unsafe')],
    });
    const fields = buildTraceSearchFields(
      [keyedField('tag'), keyedField('metadata'), keyedField('assessment')],
      [trace],
    );

    expect(fields.filter((field) => field.field.id === 'tag').map((field) => field.key)).toEqual([
      'good key',
      'good.key',
    ]);
    expect(fields.filter((field) => field.field.id === 'metadata').map((field) => field.key)).toEqual(['owner']);
    expect(fields.filter((field) => field.field.id === 'assessment').map((field) => field.key)).toEqual([
      'judge.score',
    ]);
    expect(fields.filter((field) => field.field.id === 'tag').map((field) => field.label)).toEqual([
      "tag.'good key'",
      'tag.good.key',
    ]);
  });

  test('does not fabricate a cost field when it is absent from the model', () => {
    const fields = buildTraceSearchFields([textField('duration')], [makeTrace('trace')]);
    expect(fields.some((field) => field.id === 'cost')).toBe(false);
  });
});

describe('getTraceSearchExamples', () => {
  test('ranks observed examples and uses >= for duration only when supported', () => {
    const fields = buildTraceSearchFields(
      [
        {
          ...textField('duration', [FilterOp.EQUALS, FilterOp.GREATER_THAN_OR_EQUALS]),
          valueInput: 'number',
        },
        keyedField('tag'),
        keyedField('assessment'),
        textField('session'),
        textField('trace_name'),
        textField('state'),
        textField('service_name'),
        textField('source'),
        textField('user'),
      ],
      [
        makeTrace('trace', {
          execution_duration: '1s',
          tags: { 'mlflow.traceName': 'chat', env: 'prod' },
          trace_metadata: {
            'mlflow.trace.session': 'session',
            'mlflow.trace.user': 'alice',
            'mlflow.source.name': 'app.py',
          },
          assessments: [makeFeedbackAssessment('quality', 'yes')],
        }),
      ],
    );
    const examples = getTraceSearchExamples(fields);

    expect(examples).toHaveLength(5);
    expect(examples[0].clause).toEqual({
      field: 'tag',
      operator: FilterOp.EQUALS,
      value: 'prod',
      key: 'env',
    });
    expect(examples[1].clause).toEqual({
      field: 'duration',
      operator: FilterOp.GREATER_THAN_OR_EQUALS,
      value: '1000',
    });
    expect(examples[2].clause).toEqual({
      field: 'assessment',
      operator: FilterOp.EQUALS,
      value: 'yes',
      key: 'quality',
    });
    expect(examples.map((example) => example.clause.field)).toEqual([
      'tag',
      'duration',
      'assessment',
      'state',
      'trace_name',
    ]);
    expect(examples.map((example) => example.label)).toEqual([
      "tag.env = 'prod'",
      'latency >= 1000 ms',
      "feedback.quality = 'yes'",
      "state = 'OK'",
      "trace_name = 'chat'",
    ]);
  });

  test('does not suggest a duration example when no range operator is supported', () => {
    const fields = buildTraceSearchFields(
      [{ ...textField('duration', [FilterOp.EQUALS]), valueInput: 'number' }],
      [makeTrace('trace', { execution_duration: '2s' })],
    );
    expect(fields[0].field.operators).toEqual([]);
    expect(getTraceSearchExamples(fields)).toEqual([]);
  });

  test('caps examples at five and skips empty or unsafe fields', () => {
    const fields = buildTraceSearchFields(
      [
        textField('duration'),
        textField('session'),
        textField('trace_name'),
        textField('state'),
        textField('service_name'),
        textField('source'),
        textField('user'),
        keyedField('tag'),
      ],
      [
        makeTrace('trace', {
          execution_duration: 'bad',
          tags: { 'bad`key': 'unsafe', 'mlflow.traceName': 'chat' },
          trace_metadata: {
            'mlflow.trace.session': 'session',
            'mlflow.trace.user': 'alice',
            'mlflow.source.name': 'app.py',
          },
        }),
      ],
    );

    const examples = getTraceSearchExamples(fields);
    expect(examples).toHaveLength(5);
    expect(examples.some((example) => example.field.key === 'bad`key')).toBe(false);
    expect(examples.every((example) => example.clause.value !== '')).toBe(true);
  });

  test('returns no examples for non-finite or negative durations', () => {
    const fields = buildTraceSearchFields(
      [
        {
          ...textField('duration', [FilterOp.GREATER_THAN_OR_EQUALS]),
          valueInput: 'number',
        },
      ],
      [makeTrace('negative', { execution_duration: '-1s' }), makeTrace('unknown', { execution_duration: 'oops' })],
    );
    expect(fields[0].values).toEqual([]);
    expect(getTraceSearchExamples(fields)).toEqual([]);
  });
});

describe('feedback and expectation suggestions', () => {
  test.each([false, true])('isolates same-name values regardless of row order (reversed: %s)', (reverse) => {
    const traces = [
      makeTrace('feedback', {
        assessments: [makeFeedbackAssessment('quality', 'pass')],
      }),
      makeTrace('expectation', {
        assessments: [makeExpectation('quality', 'gold')],
        trace_metadata: { unrelated: 'metadata-only' },
      }),
    ];
    const fields = buildTraceSearchFields(
      [keyedField('assessment'), keyedField('expectation')],
      reverse ? [...traces].reverse() : traces,
    );
    expect(fields).toHaveLength(2);
    expect(new Set(fields.map(({ id }) => id)).size).toBe(2);
    expect(fields.map(({ label, values }) => ({ label, values }))).toEqual([
      {
        label: 'feedback.quality',
        values: [{ value: 'pass', label: 'pass' }],
      },
      {
        label: 'expectation.quality',
        values: [{ value: 'gold', label: 'gold' }],
      },
    ]);
    const examples = getTraceSearchExamples(fields);
    expect(examples.map(({ label }) => label).sort()).toEqual([
      "expectation.quality = 'gold'",
      "feedback.quality = 'pass'",
    ]);
    expect(compileFilterModel(examples.map(({ clause }) => clause)).sort()).toEqual([
      "expectation.`quality` = 'gold'",
      "feedback.`quality` = 'pass'",
    ]);
    expect(fields.every(({ label, aliases }) => !/assessment/i.test([label, ...aliases].join(' ')))).toBe(true);
  });

  test('keeps numeric names but offers no numeric values, which may be stored as 4 or 4.0', () => {
    const fields = buildTraceSearchFields(
      [keyedField('assessment')],
      [
        makeTrace('a', { assessments: [makeFeedbackAssessment('relevance', 4)] }),
        makeTrace('b', { assessments: [makeFeedbackAssessment('relevance', 'n/a')] }),
        makeTrace('c', { assessments: [makeFeedbackAssessment('score', 0.5)] }),
      ],
    );
    expect(fields.map(({ key, values }) => ({ key, values }))).toEqual([
      { key: 'relevance', values: [{ value: 'n/a', label: 'n/a' }] },
      { key: 'score', values: [] },
    ]);
  });

  test('preserves false expectation values and excludes invalid, missing, or unsafe values', () => {
    const fields = buildTraceSearchFields(
      [keyedField('expectation')],
      [
        makeTrace('expectations', {
          assessments: [
            makeExpectation('boolean', false),
            makeExpectation('score', 0),
            { ...makeExpectation('invalid', 'skip'), valid: false },
            {
              ...makeExpectation('missing', 'skip'),
              expectation: { serialized_value: { value: '', serialization_format: 'json' } },
            },
            makeExpectation('bad`name', 'skip'),
            makeFeedbackAssessment('feedback-only', 'skip'),
          ],
        }),
      ],
    );
    expect(fields.map(({ key, values }) => ({ key, values }))).toEqual([
      { key: 'boolean', values: [{ value: 'false', label: 'false' }] },
      { key: 'score', values: [] },
    ]);
  });
});
