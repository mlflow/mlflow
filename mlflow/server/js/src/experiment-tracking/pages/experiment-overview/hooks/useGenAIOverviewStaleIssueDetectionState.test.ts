import { describe, expect, test } from '@jest/globals';
import { AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import {
  GENAI_OVERVIEW_STALE_ISSUE_DETECTION_DAY_COUNT,
  resolveGenAIOverviewStaleIssueDetectionState,
  type ResolveGenAIOverviewStaleIssueDetectionStateParams,
} from './useGenAIOverviewStaleIssueDetectionState';

const DAY_IN_MILLISECONDS = 24 * 60 * 60 * 1000;
const nowMs = Date.UTC(2026, 9, 4);

const resolveState = (overrides: Partial<ResolveGenAIOverviewStaleIssueDetectionStateParams> = {}) =>
  resolveGenAIOverviewStaleIssueDetectionState({
    enabled: true,
    latestRunUuid: 'run-1',
    latestRunCompletedAtMs: nowMs - GENAI_OVERVIEW_STALE_ISSUE_DETECTION_DAY_COUNT * DAY_IN_MILLISECONDS,
    nowMs,
    dataPoints: [
      {
        metric_name: TraceMetricKey.TRACE_COUNT,
        values: { [AggregationType.COUNT]: 12 },
        dimensions: {},
      },
    ],
    isLoading: false,
    error: null,
    ...overrides,
  });

describe('resolveGenAIOverviewStaleIssueDetectionState', () => {
  test('returns the elapsed days and new trace count at the stale threshold', () => {
    expect(resolveState()).toEqual({
      status: 'ready',
      latestRunUuid: 'run-1',
      elapsedDays: GENAI_OVERVIEW_STALE_ISSUE_DETECTION_DAY_COUNT,
      newTraceCount: 12,
    });
  });

  test.each([
    { enabled: false },
    { latestRunUuid: undefined },
    { latestRunCompletedAtMs: nowMs - (GENAI_OVERVIEW_STALE_ISSUE_DETECTION_DAY_COUNT - 1) * DAY_IN_MILLISECONDS },
    { dataPoints: [] },
    { error: new Error('failed') },
  ])('stays hidden when the reminder is not actionable', (overrides) => {
    expect(resolveState(overrides)).toEqual({ status: 'hidden' });
  });

  test('keeps the state loading while the new trace count is unresolved', () => {
    expect(resolveState({ isLoading: true, dataPoints: undefined })).toEqual({ status: 'loading' });
  });
});
