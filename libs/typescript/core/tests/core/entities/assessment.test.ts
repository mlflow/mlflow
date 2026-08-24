import {
  AssessmentSourceType,
  Feedback,
  STACK_TRACE_TRUNCATION_LENGTH,
  STACK_TRACE_TRUNCATION_PREFIX,
  assessmentFromJson,
  assessmentToJson,
  assertFeedbackPayload,
  isFeedback,
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

  it('leaves expectation assessments opaque so TraceInfo does not drop them', () => {
    const raw = {
      assessment_name: 'expected_response',
      expectation: { value: 'Paris' },
    };
    const parsed = assessmentFromJson(raw);
    expect(parsed).toEqual(raw);
    expect(assessmentToJson(parsed)).toEqual(raw);
  });

  it('exports isFeedback for public Assessment narrowing', () => {
    const feedback = new Feedback({ name: 'correctness', value: true });
    const opaque = { assessment_name: 'expected_response', expectation: { value: 'Paris' } };
    expect(isFeedback(feedback)).toBe(true);
    expect(isFeedback(opaque)).toBe(false);
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
