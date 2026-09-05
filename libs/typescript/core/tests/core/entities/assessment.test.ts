import {
  AssessmentSourceType,
  Expectation,
  Feedback,
  STACK_TRACE_TRUNCATION_LENGTH,
  STACK_TRACE_TRUNCATION_PREFIX,
  assessmentFromJson,
  assessmentToJson,
  assertFeedbackPayload,
  isFeedback,
  isExpectation,
  serializeV4TraceLocation,
  standardizeSourceType,
  truncateStackTrace,
} from '../../../src/core/entities/assessment';

describe('assessment entities', () => {
  it('serializes feedback to proto JSON snake_case', () => {
    const feedback = new Feedback({
      name: 'correctness',
      value: true,
      source: { sourceType: AssessmentSourceType.HUMAN, sourceId: 'reviewer@example.com' },
      rationale: 'Looks right',
      metadata: { project: 'demo' },
      traceId: 'tr-1',
    });

    expect(feedback.toJson()).toEqual({
      assessment_name: 'correctness',
      source: { source_type: 'HUMAN', source_id: 'reviewer@example.com' },
      trace_id: 'tr-1',
      rationale: 'Looks right',
      metadata: { project: 'demo' },
      feedback: { value: true },
    });
  });

  it('round-trips proto JSON including nested feedback error', () => {
    const parsed = Feedback.fromJson({
      assessment_id: 'a-1',
      assessment_name: 'faithfulness',
      trace_id: 'tr-2',
      source: { source_type: 'LLM_JUDGE', source_id: 'gpt-4' },
      feedback: {
        value: 0.9,
        error: { error_code: 'TIMEOUT', error_message: 'judge timed out' },
      },
      valid: true,
    });

    expect(parsed).toBeInstanceOf(Feedback);
    expect(parsed.value).toBe(0.9);
    expect(parsed.error?.errorCode).toBe('TIMEOUT');
    expect(assessmentToJson(parsed).feedback?.error?.error_code).toBe('TIMEOUT');
  });

  it('maps deprecated AI_JUDGE to LLM_JUDGE', () => {
    expect(standardizeSourceType('AI_JUDGE')).toBe(AssessmentSourceType.LLM_JUDGE);
  });

  it('rejects invalid source types', () => {
    expect(() => standardizeSourceType('ROBOT')).toThrow(/Invalid assessment source type/);
  });

  it('requires value or error for logFeedback payloads', () => {
    expect(() => assertFeedbackPayload({})).toThrow(/logFeedback requires/);
    expect(() => assertFeedbackPayload({ value: false })).not.toThrow();
    expect(() => assertFeedbackPayload({ error: 'failed' })).not.toThrow();
  });

  it('parses expectation assessments into typed entities', () => {
    const raw = {
      assessment_name: 'expected_response',
      source: { source_type: 'HUMAN', source_id: 'reviewer' },
      expectation: { value: 'Paris' },
    };
    const parsed = assessmentFromJson(raw);
    expect(parsed).toBeInstanceOf(Expectation);
    expect((parsed as Expectation).value).toBe('Paris');
    expect(assessmentToJson(parsed)).toEqual(raw);
  });

  it('serializes structured expectation values like the Python SDK', () => {
    const expectation = new Expectation({
      name: 'expected_response',
      value: { answer: 'Paris', citations: ['source-1'] },
    });

    expect(expectation.toJson().expectation).toEqual({
      serialized_value: {
        serialization_format: 'JSON_FORMAT',
        value: '{"answer":"Paris","citations":["source-1"]}',
      },
    });
    expect(Expectation.fromJson(expectation.toJson()).value).toEqual({
      answer: 'Paris',
      citations: ['source-1'],
    });
  });

  it('defaults expectation sources to HUMAN and rejects null values', () => {
    expect(new Expectation({ name: 'answer', value: 'Paris' }).source.sourceType).toBe('HUMAN');
    expect(() => new Expectation({ name: 'answer', value: null as never })).toThrow(
      /logExpectation requires/,
    );
  });

  it('serializes UC schema and table-prefix locations for V4 assessments', () => {
    expect(serializeV4TraceLocation('cat.sch')).toEqual({
      type: 'UC_SCHEMA',
      uc_schema: { catalog_name: 'cat', schema_name: 'sch' },
    });
    expect(serializeV4TraceLocation('cat.sch.tbl')).toEqual({
      type: 'UC_TABLE_PREFIX',
      uc_table_prefix: {
        catalog_name: 'cat',
        schema_name: 'sch',
        table_prefix: 'tbl',
      },
    });
    expect(() => serializeV4TraceLocation('cat')).toThrow(/Invalid UC location/);
  });

  it('reconstructs a V4 trace ID from an assessment response', () => {
    const parsed = Feedback.fromJson({
      assessment_name: 'correctness',
      trace_id: 'abcdef',
      trace_location: {
        type: 'UC_TABLE_PREFIX',
        uc_table_prefix: {
          catalog_name: 'cat',
          schema_name: 'sch',
          table_prefix: 'tbl',
        },
      },
      feedback: { value: true },
    });
    expect(parsed.traceId).toBe('trace:/cat.sch.tbl/abcdef');
  });

  it('exports isFeedback for public Assessment narrowing', () => {
    const feedback = new Feedback({ name: 'correctness', value: true });
    const expectation = new Expectation({ name: 'expected_response', value: 'Paris' });
    expect(isFeedback(feedback)).toBe(true);
    expect(isFeedback(expectation)).toBe(false);
    expect(isExpectation(expectation)).toBe(true);
  });

  it('maps string error overload to ASSESSMENT_ERROR like Python Feedback', () => {
    const feedback = new Feedback({ name: 'correctness', error: 'judge failed' });
    expect(feedback.error).toEqual({
      errorCode: 'ASSESSMENT_ERROR',
      errorMessage: 'judge failed',
    });
    expect(feedback.toJson().feedback?.error).toEqual({
      error_code: 'ASSESSMENT_ERROR',
      error_message: 'judge failed',
    });
  });

  it('truncates stackTrace before serialization like Python AssessmentError.to_proto', () => {
    const longTrace = 'x'.repeat(STACK_TRACE_TRUNCATION_LENGTH + 50);
    const feedback = new Feedback({
      name: 'correctness',
      error: { errorCode: 'TIMEOUT', errorMessage: 'boom', stackTrace: longTrace },
    });
    const serialized = feedback.toJson().feedback?.error?.stack_trace;
    expect(serialized).toBeDefined();
    expect(serialized!.length).toBe(STACK_TRACE_TRUNCATION_LENGTH);
    expect(serialized!.startsWith(STACK_TRACE_TRUNCATION_PREFIX)).toBe(true);
    expect(truncateStackTrace('short')).toBe('short');
  });
});
