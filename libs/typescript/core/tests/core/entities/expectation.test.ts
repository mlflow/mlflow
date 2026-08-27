import {
  AssessmentSourceType,
  Expectation,
  assessmentFromJson,
  assessmentToJson,
  isExpectation,
  isFeedback,
} from '../../../src/core/entities/assessment';

describe('Expectation entity', () => {
  it('serializes expectation to proto JSON snake_case with expectation key', () => {
    const expectation = new Expectation({
      name: 'expected_answer',
      value: 'Paris',
      source: { sourceType: AssessmentSourceType.HUMAN, sourceId: 'annotator@example.com' },
      rationale: 'Verified against atlas',
      metadata: { dataset: 'geography-v1' },
      traceId: 'tr-1',
    });

    expect(expectation.toJson()).toEqual({
      assessment_name: 'expected_answer',
      source: { source_type: 'HUMAN', source_id: 'annotator@example.com' },
      trace_id: 'tr-1',
      rationale: 'Verified against atlas',
      metadata: { dataset: 'geography-v1' },
      expectation: { value: 'Paris' },
    });
  });

  it('serializes complex expectation values (objects, arrays)', () => {
    const complexValue = {
      role: 'assistant',
      content: 'The answer is 42.',
      score: 0.95,
    };

    const expectation = new Expectation({ name: 'expected_message', value: complexValue });
    const json = expectation.toJson();

    expect(json.expectation).toEqual({ value: complexValue });
    expect(json.feedback).toBeUndefined();
  });

  it('round-trips fromJson to toJson', () => {
    const parsed = Expectation.fromJson({
      assessment_id: 'a-1',
      assessment_name: 'expected_answer',
      trace_id: 'tr-2',
      source: { source_type: 'HUMAN', source_id: 'alice' },
      expectation: { value: 42 },
      rationale: 'ground truth',
      valid: true,
    });

    expect(parsed).toBeInstanceOf(Expectation);
    expect(parsed.value).toBe(42);
    expect(parsed.assessmentId).toBe('a-1');
    expect(parsed.rationale).toBe('ground truth');
    expect(parsed.valid).toBe(true);

    const json = parsed.toJson();
    expect(json.expectation).toEqual({ value: 42 });
    expect(json.feedback).toBeUndefined();
  });

  it('defaults to HUMAN source type (Python parity for ground-truth annotations)', () => {
    const expectation = new Expectation({ name: 'expected_answer', value: 'Paris' });
    expect(expectation.source.sourceType).toBe(AssessmentSourceType.HUMAN);
    expect(expectation.source.sourceId).toBe('default');
  });

  it('isExpectation returns true for Expectation, false for Feedback and opaque objects', () => {
    const exp = new Expectation({ name: 'expected_answer', value: 'Paris' });
    const opaque = { assessment_name: 'feedback_item', feedback: { value: true } };

    expect(isExpectation(exp)).toBe(true);
    expect(isFeedback(exp)).toBe(false);
    expect(isExpectation(opaque)).toBe(false);
  });

  it('assessmentFromJson dispatches expectation key to Expectation instance', () => {
    const raw = {
      assessment_name: 'expected_response',
      expectation: { value: 'Paris' },
    };

    const parsed = assessmentFromJson(raw);
    expect(parsed).toBeInstanceOf(Expectation);
    expect((parsed as Expectation).value).toBe('Paris');
  });

  it('assessmentFromJson keeps unknown payloads opaque', () => {
    const raw = {
      assessment_name: 'some_issue',
      issue: { issue_id: 'ISS-001' },
    };

    const parsed = assessmentFromJson(raw);
    expect(parsed).toEqual(raw);
    expect(assessmentToJson(parsed)).toEqual(raw);
  });

  it('assessmentToJson round-trips an Expectation via toJson', () => {
    const exp = new Expectation({
      name: 'expected_answer',
      value: true,
      source: { sourceType: 'CODE', sourceId: 'eval-script' },
    });

    const json = assessmentToJson(exp);
    expect(json.expectation).toEqual({ value: true });
    expect(json.feedback).toBeUndefined();
  });

  it('preserves absent optional fields (no spurious keys in output)', () => {
    const expectation = new Expectation({ name: 'minimal', value: 'hello' });
    const json = expectation.toJson();

    expect(json.assessment_id).toBeUndefined();
    expect(json.trace_id).toBeUndefined();
    expect(json.span_id).toBeUndefined();
    expect(json.rationale).toBeUndefined();
    expect(json.metadata).toBeUndefined();
    expect(json.feedback).toBeUndefined();
  });

  it('maps deprecated AI_JUDGE to LLM_JUDGE in source', () => {
    const exp = new Expectation({
      name: 'expected_answer',
      value: 'Paris',
      source: { sourceType: 'AI_JUDGE' as 'LLM_JUDGE', sourceId: 'judge' },
    });
    expect(exp.source.sourceType).toBe(AssessmentSourceType.LLM_JUDGE);
  });

  it('rejects invalid source types', () => {
    expect(
      () =>
        new Expectation({
          name: 'test',
          value: 'x',
          source: { sourceType: 'ROBOT' as 'HUMAN', sourceId: 'bot' },
        }),
    ).toThrow(/Invalid assessment source type/);
  });
});
