import { describe, expect, it } from '@jest/globals';
import { createIntl } from 'react-intl';
import {
  ASSESSMENT_SESSION_METADATA_KEY,
  type Assessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  FilterOp,
  type FilterClause,
  type FilterFieldDef,
  type TraceComparisonRow,
} from '@databricks/web-shared/traces-table';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { buildEvalRunTraceComparisonRows } from './evalRunTraceComparisonRows';
import {
  filterEvalRunComparisonRows,
  getComparisonTraceInput,
  getComparisonTraceOutput,
  validateComparisonFilters,
} from './evalRunComparisonTraceFilters';

const clause = (field: string, value = '', operator = FilterOp.EQUALS, key?: string): FilterClause => ({
  field,
  value,
  operator,
  key,
});
const group = (...traces: (ModelTraceInfoV3 | undefined)[]): TraceComparisonRow => ({
  id: 'group',
  input: 'shared',
  runs: traces.map((trace, index) => ({ trace, runUuid: `run-${index}`, label: `Run ${index}`, color: 'blue' })),
});
const matches = (trace: ModelTraceInfoV3, filter: FilterClause) =>
  filterEvalRunComparisonRows([group(trace)], [filter], '').length === 1;

describe('filterEvalRunComparisonRows', () => {
  it('requires search and every condition on the same trace, retaining complete matched duplicates', () => {
    const baseline = [
      makeTrace('a1', { request_preview: 'same', state: 'OK', tags: { env: 'prod' } }),
      makeTrace('a2', { request_preview: 'same', state: 'ERROR', tags: { env: 'prod' } }),
    ];
    const candidate = [
      makeTrace('b1', { request_preview: 'same', state: 'ERROR', tags: { env: 'dev' } }),
      makeTrace('b2', { request_preview: 'same', state: 'OK', tags: { env: 'prod' } }),
    ];
    const rows = buildEvalRunTraceComparisonRows(
      [
        { runUuid: 'a', traceInfos: baseline },
        { runUuid: 'b', traceInfos: candidate },
      ],
      new Map(),
      () => 'blue',
      '',
    );
    const filters = [clause('state', 'ERROR'), clause('tag', 'prod', FilterOp.EQUALS, 'env')];
    expect(
      filterEvalRunComparisonRows(rows, filters, '').map((row) => row.runs.map(({ trace }) => trace?.trace_id)),
    ).toEqual([['a2', 'b2']]);
    expect(filterEvalRunComparisonRows(rows, filters, 'b2')).toEqual([]);
    expect(filterEvalRunComparisonRows(rows, filters, 'a2')).toEqual([rows[1]]);
    expect(filterEvalRunComparisonRows(rows, [], '')).toEqual(rows);
  });

  it('never treats an absent run slot as a null match', () => {
    const row = group(undefined, makeTrace('present', { tags: { env: 'prod' } }));
    expect(filterEvalRunComparisonRows([row], [clause('tag', 'dev', FilterOp.NOT_EQUALS, 'missing')], '')).toEqual([]);
    expect(filterEvalRunComparisonRows([group(undefined)], [], '')).toEqual([]);
    expect(filterEvalRunComparisonRows([], [], '')).toEqual([]);
  });

  it.each([
    clause('state', 'OK'),
    clause('trace_name', 'agent'),
    clause('user', 'alice'),
    clause('session', 'session'),
    clause('run_name', 'run-0'),
    clause('source', 'app'),
    clause('tag', 'prod', FilterOp.EQUALS, 'env'),
    clause('metadata', 'Root', FilterOp.EQUALS, 'mlflow.trace.service'),
    clause('tag', 'DEV', FilterOp.NOT_EQUALS, 'env'),
    clause('source', 'APP', FilterOp.CONTAINS),
  ])('matches loaded field $field $operator $value', (filter) => {
    const trace = makeTrace('a', {
      tags: { 'mlflow.traceName': 'agent', env: 'prod' },
      trace_metadata: {
        'mlflow.trace.service': 'Root',
        'mlflow.trace.user': 'alice',
        'mlflow.trace.session': 'session',
        'mlflow.source.name': 'app',
        'mlflow.sourceRun': 'original',
      },
    });
    expect(matches(trace, filter)).toBe(true);
    expect(matches(trace, clause('run_name', 'original'))).toBe(false);
  });

  it.each([
    [FilterOp.EQUALS, '1200', true],
    [FilterOp.NOT_EQUALS, '1200', false],
    [FilterOp.GREATER_THAN, '1000', true],
    [FilterOp.GREATER_THAN_OR_EQUALS, '1200', true],
    [FilterOp.LESS_THAN, '1000', false],
    [FilterOp.LESS_THAN_OR_EQUALS, '1200', true],
  ])('compares duration in milliseconds with %s', (operator, value, expected) => {
    expect(matches(makeTrace('a'), clause('duration', value, operator))).toBe(expected);
    expect(matches(makeTrace('a', { execution_duration: undefined }), clause('duration', value, operator))).toBe(false);
  });

  it('uses latest eligible trace-level assessment of the requested kind', () => {
    const expected: Assessment = {
      ...makeFeedbackAssessment('quality', 0),
      feedback: undefined,
      expectation: { value: 10 },
    };
    const trace = makeTrace('a', {
      assessments: [
        makeFeedbackAssessment('quality', 1),
        makeFeedbackAssessment('quality', 0, { create_time: '2026-01-01' }),
        makeFeedbackAssessment('quality', 5, { valid: false, create_time: '2027-01-01' }),
        makeFeedbackAssessment('quality', 8, {
          metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session' },
          create_time: '2028-01-01',
        }),
        expected,
      ],
    });
    expect(matches(trace, clause('assessment', '0.0', FilterOp.EQUALS, 'quality'))).toBe(true);
    expect(matches(trace, clause('expectation', '10', FilterOp.EQUALS, 'quality'))).toBe(true);
    expect(matches(trace, clause('assessment', '00', FilterOp.EQUALS, 'quality'))).toBe(false);
  });

  it.each([
    { value: 0, filter: '0', op: FilterOp.EQUALS, expected: true },
    { value: false, filter: 'false', op: FilterOp.EQUALS, expected: true },
    { value: '01.0', filter: '1', op: FilterOp.EQUALS, expected: true },
    { value: 0, filter: '1', op: FilterOp.NOT_EQUALS, expected: true },
    { value: false, filter: 'true', op: FilterOp.NOT_EQUALS, expected: true },
  ])('distinguishes assessment $value with $op $filter', ({ value, filter, op, expected }) => {
    expect(
      matches(
        makeTrace('a', { assessments: [makeFeedbackAssessment('quality', value)] }),
        clause('assessment', filter, op, 'quality'),
      ),
    ).toBe(expected);
  });

  it.each(['assessment', 'feedback'])(
    'matches a value-less %s error only with the explicit error filter',
    (errorLocation) => {
      const error = { error_code: 'FAILED', error_message: 'failure' };
      const failed = makeTrace('a', {
        assessments: [
          makeFeedbackAssessment(
            'quality',
            0,
            errorLocation === 'assessment' ? { feedback: {}, error } : { feedback: { error } },
          ),
        ],
      });
      expect(matches(failed, clause('assessment', 'Error', FilterOp.EQUALS, 'quality'))).toBe(true);
      expect(matches(failed, clause('assessment', '3', FilterOp.NOT_EQUALS, 'quality'))).toBe(false);
      expect(matches(failed, clause('assessment', '0', FilterOp.EQUALS, 'quality'))).toBe(false);
    },
  );

  it('does not match a missing assessment with an error or inequality filter', () => {
    const missing = makeTrace('missing');
    expect(matches(missing, clause('assessment', 'Error', FilterOp.EQUALS, 'quality'))).toBe(false);
    expect(matches(missing, clause('assessment', '3', FilterOp.NOT_EQUALS, 'quality'))).toBe(false);
  });

  it('matches case-sensitive regex against the available raw payload without ignoring invalid regex', () => {
    const row = group(makeTrace('a', { request: '{"Full":"value"}', request_preview: 'preview' }));
    expect(filterEvalRunComparisonRows([row], [clause('input', 'Full.*value', FilterOp.RLIKE)], '')).toEqual([row]);
    expect(filterEvalRunComparisonRows([row], [clause('input', 'full', FilterOp.RLIKE)], '')).toEqual([]);
    expect(filterEvalRunComparisonRows([row], [clause('input', '[', FilterOp.RLIKE)], '')).toEqual([]);
  });
});

describe('available comparison payloads', () => {
  it('prefers recorded payload then request/response, metadata and finally preview, preserving empty strings', () => {
    const trace = makeTrace('a', {
      request: 'request',
      response: 'response',
      trace_metadata: { 'mlflow.traceInputs': 'metadata-in', 'mlflow.traceOutputs': 'metadata-out' },
    });
    expect(getComparisonTraceInput(trace, new Map([['a', 'recorded']]))).toBe('recorded');
    expect(getComparisonTraceInput(trace)).toBe('request');
    expect(getComparisonTraceInput({ ...trace, request: undefined })).toBe('metadata-in');
    expect(getComparisonTraceInput({ ...trace, request: undefined, trace_metadata: {} })).toBe('request for a');
    expect(getComparisonTraceInput(trace, new Map([['a', '']]))).toBe('');
    expect(getComparisonTraceOutput(trace)).toBe('response');
    expect(getComparisonTraceOutput({ ...trace, response: undefined })).toBe('metadata-out');
    expect(getComparisonTraceOutput({ ...trace, response: undefined, trace_metadata: {} })).toBe('response for a');
    expect(getComparisonTraceOutput({ ...trace, response: '' })).toBe('');
  });
});

describe('validateComparisonFilters', () => {
  const intl = createIntl({ locale: 'en' });
  const fields: FilterFieldDef[] = [
    { id: 'duration', label: 'Duration', operators: [FilterOp.EQUALS], valueInput: 'number' },
    { id: 'input', label: 'Input', operators: [FilterOp.RLIKE], valueInput: 'text' },
    { id: 'tag', label: 'Tag', operators: [FilterOp.EQUALS], valueInput: 'text', requiresKey: true },
  ];
  it.each([
    clause('span_name', 'ignored'),
    clause('duration', 'Infinity'),
    clause('duration', 'invalid'),
    clause('duration', ''),
    clause('duration', '2', FilterOp.CONTAINS),
    clause('tag', 'prod', FilterOp.EQUALS),
    clause('input', '[', FilterOp.RLIKE),
    clause('input', '\\Ahello', FilterOp.RLIKE),
    clause('input', '(?i)hello', FilterOp.RLIKE),
  ])('rejects invalid $field $operator $value', (filter) =>
    expect(validateComparisonFilters([filter], fields, intl)).toEqual(expect.any(String)),
  );
  it('accepts supported finite numeric values, keyed clauses and unicode regex', () => {
    expect(
      validateComparisonFilters(
        [
          clause('duration', '1e3'),
          clause('tag', 'prod', FilterOp.EQUALS, 'env'),
          clause('input', '你好.*', FilterOp.RLIKE),
        ],
        fields,
        intl,
      ),
    ).toBeUndefined();
  });
});
