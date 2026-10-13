import { describe, test, expect } from '@jest/globals';
import { getPreservedQueryString, isTracesRelatedTab } from './utils';
import { ExperimentPageTabName } from '../../../constants';

describe('getPreservedQueryString', () => {
  test('preserves time range params', () => {
    const result = getPreservedQueryString('?startTimeLabel=LAST_24_HOURS&startTime=2024-01-01&endTime=2024-01-02');
    expect(result).toContain('startTimeLabel=LAST_24_HOURS');
    expect(result).toContain('startTime=2024-01-01');
    expect(result).toContain('endTime=2024-01-02');
  });

  test('preserves workflowType param', () => {
    const result = getPreservedQueryString('?workflowType=genai&startTimeLabel=LAST_24_HOURS');
    expect(result).toContain('workflowType=genai');
    expect(result).toContain('startTimeLabel=LAST_24_HOURS');
  });

  test('preserves workspace param', () => {
    const result = getPreservedQueryString('?workspace=default&startTimeLabel=LAST_24_HOURS&workflowType=genai');
    expect(result).toContain('workspace=default');
    expect(result).toContain('startTimeLabel=LAST_24_HOURS');
    expect(result).toContain('workflowType=genai');
  });

  test('strips non-preserved params', () => {
    const result = getPreservedQueryString(
      '?startTimeLabel=LAST_24_HOURS&selectedTraceId=trace-123&viewMode=list&workflowType=machine_learning&workspace=default',
    );
    expect(result).toContain('startTimeLabel=LAST_24_HOURS');
    expect(result).toContain('workflowType=machine_learning');
    expect(result).toContain('workspace=default');
    expect(result).not.toContain('selectedTraceId');
    expect(result).not.toContain('viewMode');
  });

  test('returns undefined when no preserved params exist', () => {
    expect(getPreservedQueryString('?selectedTraceId=trace-123')).toBeUndefined();
    expect(getPreservedQueryString('')).toBeUndefined();
  });
});

describe('isTracesRelatedTab', () => {
  test('returns true for traces-related tabs', () => {
    expect(isTracesRelatedTab(ExperimentPageTabName.Overview)).toBe(true);
    expect(isTracesRelatedTab(ExperimentPageTabName.Traces)).toBe(true);
    expect(isTracesRelatedTab(ExperimentPageTabName.ChatSessions)).toBe(true);
    expect(isTracesRelatedTab(ExperimentPageTabName.SingleChatSession)).toBe(true);
  });

  test('returns false for non-traces-related tabs', () => {
    expect(isTracesRelatedTab(ExperimentPageTabName.Runs)).toBe(false);
    expect(isTracesRelatedTab(ExperimentPageTabName.Models)).toBe(false);
    expect(isTracesRelatedTab(ExperimentPageTabName.Datasets)).toBe(false);
  });
});
