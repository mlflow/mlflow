import { describe, expect, test } from '@jest/globals';

import { getUniqueIssueReferenceAssessments, getUniqueIssueReferences } from './IssueAssessmentUtils';
import type { Assessment, IssueReferenceAssessment } from './ModelTrace.types';

const reference = (
  overrides: Partial<IssueReferenceAssessment> & Pick<IssueReferenceAssessment, 'assessment_id'>,
): IssueReferenceAssessment => ({
  assessment_name: 'iss-latency',
  trace_id: 'trace-1',
  span_id: 'span-1',
  source: { source_type: 'LLM_JUDGE', source_id: 'mlflow.issue_detection' },
  create_time: '2026-10-05T05:35:58Z',
  last_update_time: '2026-10-05T05:35:58Z',
  issue: { issue_name: 'High latency' },
  rationale: 'The tool took too long.',
  ...overrides,
});

describe('getUniqueIssueReferenceAssessments', () => {
  test('a rerun over the same trace leaves one reference per issue, the newest', () => {
    // Each detection run writes its own reference, so a reader taking them as they come showed a
    // card per run and an issue count that grew with every rerun.
    const first = reference({ assessment_id: 'a-first', rationale: 'From the first run.' });
    const second = reference({
      assessment_id: 'a-second',
      last_update_time: '2026-10-06T05:35:58Z',
      rationale: 'From the rerun.',
    });

    const unique = getUniqueIssueReferenceAssessments([first, second]);

    expect(unique).toHaveLength(1);
    expect(unique[0].rationale).toBe('From the rerun.');
  });

  test('order does not decide it; the timestamp does', () => {
    const newer = reference({ assessment_id: 'a-newer', last_update_time: '2026-10-06T05:35:58Z' });
    const older = reference({ assessment_id: 'a-older', last_update_time: '2026-10-05T05:35:58Z' });

    expect(getUniqueIssueReferenceAssessments([newer, older])[0].assessment_id).toBe('a-newer');
    expect(getUniqueIssueReferenceAssessments([older, newer])[0].assessment_id).toBe('a-newer');
  });

  test('an invalidated reference is not an association', () => {
    const invalidated = reference({ assessment_id: 'a-invalid', valid: false });

    expect(getUniqueIssueReferenceAssessments([invalidated])).toEqual([]);
  });

  test('references to different issues are all kept', () => {
    const latency = reference({ assessment_id: 'a-latency' });
    const cost = reference({ assessment_id: 'a-cost', assessment_name: 'iss-cost' });

    expect(getUniqueIssueReferenceAssessments([latency, cost])).toHaveLength(2);
  });

  test('ordinary feedback is never read as an issue association', () => {
    const feedback: Assessment = {
      assessment_id: 'a-feedback',
      assessment_name: 'quality',
      trace_id: 'trace-1',
      span_id: 'span-1',
      source: { source_type: 'LLM_JUDGE', source_id: 'mlflow.issue_detection' },
      create_time: '2026-10-05T05:35:58Z',
      last_update_time: '2026-10-05T05:35:58Z',
      feedback: { value: true },
    };

    expect(getUniqueIssueReferenceAssessments([feedback])).toEqual([]);
  });

  test('the projected form selects the same references', () => {
    // The banner and the assessments pane take different shapes of the same selection, so the
    // rule lives in one place and this pins the two views to it.
    const first = reference({ assessment_id: 'a-first' });
    const second = reference({ assessment_id: 'a-second', last_update_time: '2026-10-06T05:35:58Z' });
    const other = reference({ assessment_id: 'a-other', assessment_name: 'iss-cost' });
    const assessments = [first, second, other];

    expect(getUniqueIssueReferences(assessments).map(({ id }) => id)).toEqual(
      getUniqueIssueReferenceAssessments(assessments).map(({ assessment_name }) => assessment_name),
    );
  });
});
