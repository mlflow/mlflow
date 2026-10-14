import { describe, expect, test } from '@jest/globals';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import type { RunEntity, RunInfoEntity } from '../../../types';
import type { Issue, IssueSeverity, IssueStatus } from '../../../components/run-page/hooks/useSearchIssuesQuery';
import {
  findLatestReadableIssueDetectionRun,
  resolveGenAIOverviewAnalyzeState,
  type LatestReadableIssueDetectionRunDependencies,
} from './useGenAIOverviewAnalyzeState';

const DAY_IN_MILLISECONDS = 24 * 60 * 60 * 1000;
const startTimeMs = Date.UTC(2026, 7, 1);
const endTimeMs = startTimeMs + 2 * DAY_IN_MILLISECONDS;

const createRun = (status: RunInfoEntity['status'], runUuid: string, startTime: number, endTime = 0): RunEntity => ({
  info: {
    artifactUri: '',
    endTime,
    experimentId: 'experiment-1',
    lifecycleStage: 'active',
    runUuid,
    runName: 'Issue detection',
    startTime,
    status,
  },
  data: { tags: [], params: [], metrics: [] },
});

const createIssue = (issueId: string, severity?: IssueSeverity, status: IssueStatus = 'pending'): Issue => ({
  issue_id: issueId,
  experiment_id: 'experiment-1',
  name: `Issue ${issueId}`,
  severity,
  status,
  created_timestamp: startTimeMs,
  last_updated_timestamp: startTimeMs,
});

describe('resolveGenAIOverviewAnalyzeState', () => {
  test('aggregates visible OSS issues by day and severity', () => {
    const firstRun = createRun('FINISHED', 'run-1', startTimeMs);
    const latestRun = createRun('FINISHED', 'run-2', startTimeMs + DAY_IN_MILLISECONDS);

    expect(
      resolveGenAIOverviewAnalyzeState({
        runs: [latestRun, firstRun],
        isLoading: false,
        error: null,
        issuesByRunUuid: new Map([
          ['run-1', [createIssue('high', 'high'), createIssue('rejected', 'low', 'rejected')]],
          ['run-2', [createIssue('unclassified'), createIssue('not-an-issue', 'not_an_issue')]],
        ]),
        areIssuesLoading: false,
        startTimeMs,
        endTimeMs,
        recentIssueStartTimeMs: startTimeMs,
        latestReadableRun: { runUuid: 'run-2', completedAtMs: startTimeMs + DAY_IN_MILLISECONDS },
        isLatestReadableRunLoading: false,
        latestReadableRunError: null,
      }),
    ).toEqual({
      status: 'completed',
      runUuid: 'run-2',
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
});

describe('findLatestReadableIssueDetectionRun', () => {
  test('skips a missing result and continues across search pages', async () => {
    const fetchedRunUuids: string[] = [];
    const dependencies: LatestReadableIssueDetectionRunDependencies = {
      searchRuns: async (_experimentId, pageToken) =>
        pageToken
          ? { runs: [createRun('FINISHED', 'run-readable', startTimeMs)] }
          : {
              runs: [createRun('FINISHED', 'run-missing', endTimeMs)],
              next_page_token: 'next-page',
            },
      fetchIssues: async (_experimentId, runUuid) => {
        fetchedRunUuids.push(runUuid);
        if (runUuid === 'run-missing') throw new ErrorWrapper('missing', 404);
        return [];
      },
    };

    await expect(findLatestReadableIssueDetectionRun('experiment-1', dependencies)).resolves.toEqual({
      runUuid: 'run-readable',
      completedAtMs: startTimeMs,
    });
    expect(fetchedRunUuids).toEqual(['run-missing', 'run-readable']);
  });

  test('propagates transient issue-search failures', async () => {
    const requestError = new ErrorWrapper('temporarily unavailable', 503);
    const dependencies: LatestReadableIssueDetectionRunDependencies = {
      searchRuns: async () => ({ runs: [createRun('FINISHED', 'run-1', startTimeMs)] }),
      fetchIssues: async () => {
        throw requestError;
      },
    };

    await expect(findLatestReadableIssueDetectionRun('experiment-1', dependencies)).rejects.toBe(requestError);
  });
});
