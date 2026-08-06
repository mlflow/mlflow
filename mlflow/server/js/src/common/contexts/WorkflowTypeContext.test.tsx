import { describe, test, expect } from '@jest/globals';
import { getWorkflowTypeFromUrl, buildWorkflowTypeQueryString, WorkflowType } from './WorkflowTypeContext';

describe('getWorkflowTypeFromUrl', () => {
  test('returns genai when URL has workflowType=genai', () => {
    const params = new URLSearchParams('workflowType=genai');
    expect(getWorkflowTypeFromUrl(params, WorkflowType.MACHINE_LEARNING)).toBe(WorkflowType.GENAI);
  });

  test('returns machine_learning when URL has workflowType=machine_learning', () => {
    const params = new URLSearchParams('workflowType=machine_learning');
    expect(getWorkflowTypeFromUrl(params, WorkflowType.GENAI)).toBe(WorkflowType.MACHINE_LEARNING);
  });

  test('returns fallback when URL has no workflowType param', () => {
    const params = new URLSearchParams('');
    expect(getWorkflowTypeFromUrl(params, WorkflowType.GENAI)).toBe(WorkflowType.GENAI);
    expect(getWorkflowTypeFromUrl(params, WorkflowType.MACHINE_LEARNING)).toBe(WorkflowType.MACHINE_LEARNING);
  });

  test('returns fallback when URL has invalid workflowType value', () => {
    const params = new URLSearchParams('workflowType=invalid');
    expect(getWorkflowTypeFromUrl(params, WorkflowType.GENAI)).toBe(WorkflowType.GENAI);
  });

  test('ignores unrelated params', () => {
    const params = new URLSearchParams('workspace=default&startTimeLabel=LAST_24_HOURS');
    expect(getWorkflowTypeFromUrl(params, WorkflowType.MACHINE_LEARNING)).toBe(WorkflowType.MACHINE_LEARNING);
  });
});

describe('buildWorkflowTypeQueryString', () => {
  test('builds query with workflowType only', () => {
    const params = new URLSearchParams('');
    expect(buildWorkflowTypeQueryString(WorkflowType.GENAI, params)).toBe('workflowType=genai');
  });

  test('preserves workspace param', () => {
    const params = new URLSearchParams('workspace=default');
    const result = buildWorkflowTypeQueryString(WorkflowType.MACHINE_LEARNING, params);
    expect(result).toContain('workflowType=machine_learning');
    expect(result).toContain('workspace=default');
  });

  test('does not carry over unrelated params', () => {
    const params = new URLSearchParams('selectedTraceId=trace-123&startTimeLabel=LAST_24_HOURS&workspace=default');
    const result = buildWorkflowTypeQueryString(WorkflowType.GENAI, params);
    expect(result).toContain('workflowType=genai');
    expect(result).toContain('workspace=default');
    expect(result).not.toContain('selectedTraceId');
    expect(result).not.toContain('startTimeLabel');
  });
});
