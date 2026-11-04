import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import type { RunEntity, RunInfoEntity } from '../../../types';
import type { Issue, IssueSeverity, IssueStatus } from '../../../components/run-page/hooks/useSearchIssuesQuery';
import { fetchGenAIOverviewIssues, resolveGenAIOverviewAnalyzeState } from './useGenAIOverviewAnalyzeState';

const mockFetchAPI = jest.fn<(...args: unknown[]) => Promise<unknown>>();

jest.mock('../../../../common/utils/FetchUtils', () => ({
  fetchAPI: (...args: unknown[]) => mockFetchAPI(...args),
  getAjaxUrl: (path: string) => path,
}));

const DAY_IN_MILLISECONDS = 24 * 60 * 60 * 1000;
const startTimeMs = Date.UTC(2026, 7, 1);
const endTimeMs = startTimeMs + 2 * DAY_IN_MILLISECONDS;

const createRun = (status: RunInfoEntity['status'], startTime = startTimeMs, endTime = 0): RunEntity => ({
  info: {
    artifactUri: '',
    endTime,
    experimentId: 'experiment-1',
    lifecycleStage: 'active',
    runUuid: 'run-1',
    runName: 'Issue detection',
    startTime,
    status,
  },
  data: { tags: [], params: [], metrics: [] },
});

const createIssue = (
  issueId: string,
  createdTimestamp: number,
  severity?: IssueSeverity,
  status: IssueStatus = 'pending',
): Issue => ({
  issue_id: issueId,
  experiment_id: 'experiment-1',
  name: `Issue ${issueId}`,
  severity,
  status,
  created_timestamp: createdTimestamp,
  last_updated_timestamp: createdTimestamp,
});

const resolveState = (overrides: Partial<Parameters<typeof resolveGenAIOverviewAnalyzeState>[0]> = {}) =>
  resolveGenAIOverviewAnalyzeState({
    latestCompletedRun: undefined,
    isLoading: false,
    error: null,
    issues: [],
    startTimeMs,
    endTimeMs,
    recentIssueStartTimeMs: startTimeMs,
    ...overrides,
  });

describe('resolveGenAIOverviewAnalyzeState', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test.each([
    [{ isLoading: true }, 'loading'],
    [{ error: new Error('failed') }, 'error'],
    [{}, 'never-run'],
  ])('returns $expectedStatus', (overrides, expectedStatus) => {
    expect(resolveState(overrides)).toMatchObject({ status: expectedStatus });
  });

  test('counts issues in the displayed range and buckets them by creation date', () => {
    const latestRun = createRun('FINISHED', startTimeMs, startTimeMs + DAY_IN_MILLISECONDS);

    expect(
      resolveState({
        latestCompletedRun: latestRun,
        issues: [
          createIssue('high', startTimeMs, 'high'),
          createIssue('unclassified', startTimeMs + DAY_IN_MILLISECONDS),
          createIssue('rejected', startTimeMs, 'low', 'rejected'),
          createIssue('not-an-issue', startTimeMs, 'not_an_issue'),
        ],
      }),
    ).toEqual({
      status: 'completed',
      runUuid: 'run-1',
      latestRunCompletedAtMs: startTimeMs + DAY_IN_MILLISECONDS,
      issueCount: 2,
      issuesCreatedInLastSevenDays: 2,
      activity: [
        { timestampMs: startTimeMs, count: 1, high: 1, medium: 0, low: 0 },
        { timestampMs: startTimeMs + DAY_IN_MILLISECONDS, count: 1, high: 0, medium: 1, low: 0 },
        { timestampMs: endTimeMs, count: 0, high: 0, medium: 0, low: 0 },
      ],
    });
  });

  test('counts recent issues outside the displayed activity range', () => {
    expect(
      resolveState({
        latestCompletedRun: createRun('FINISHED'),
        issues: [createIssue('recent', endTimeMs + DAY_IN_MILLISECONDS, 'medium')],
      }),
    ).toMatchObject({
      issueCount: 0,
      issuesCreatedInLastSevenDays: 1,
      activity: expect.not.arrayContaining([expect.objectContaining({ count: 1 })]),
    });
  });
});

describe('fetchGenAIOverviewIssues', () => {
  test('loads experiment-wide visible issues in one request', async () => {
    const issue = createIssue('issue-1', startTimeMs, 'high');
    mockFetchAPI.mockResolvedValue({ issues: [issue] });

    await expect(fetchGenAIOverviewIssues('experiment-1')).resolves.toEqual({ issues: [issue] });
    expect(mockFetchAPI).toHaveBeenCalledTimes(1);
    const [, request] = mockFetchAPI.mock.calls[0];
    expect(request).toEqual({
      method: 'POST',
      signal: undefined,
      body: {
        experiment_id: 'experiment-1',
        statuses: ['pending', 'resolved'],
        severities: ['low', 'medium', 'high'],
        order_by: ['created_timestamp DESC'],
        max_results: 1000,
        view: 'basic',
      },
    });
    expect((request as { body: Record<string, unknown> }).body).not.toHaveProperty('include_trace_count');
  });
});
