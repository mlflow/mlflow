import { jest, describe, test, expect, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { act, renderWithIntl, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { IssueDetectionProgress } from './IssueDetectionProgress';
import { JobStatus } from '../hooks/useFetchJobStatus';
import { useCancelJob } from '../hooks/useCancelJob';

jest.mock('@mlflow/mlflow/src/telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: jest.fn(() => jest.fn()),
}));

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/RoutingUtils')>(
    '../../../../common/utils/RoutingUtils',
  ),
  useParams: jest.fn(() => ({ experimentId: 'test-exp', runUuid: 'test-run-uuid' })),
  useNavigate: jest.fn(() => jest.fn()),
}));

jest.mock('../hooks/useSearchIssuesQuery', () => ({
  useSearchIssuesQuery: jest.fn(() => ({ issues: [] })),
}));

jest.mock('../hooks/useCancelJob', () => ({
  useCancelJob: jest.fn(() => ({
    cancelJob: jest.fn(),
    cancelJobAsync: jest.fn(() => Promise.resolve()),
    isCancelling: false,
  })),
}));

const { useLogTelemetryEvent } = jest.requireMock('@mlflow/mlflow/src/telemetry/hooks/useLogTelemetryEvent') as {
  useLogTelemetryEvent: jest.Mock;
};

describe('IssueDetectionProgress cancel', () => {
  test('offers Cancel while running', () => {
    renderWithIntl(
      <MemoryRouter>
        <IssueDetectionProgress jobStatus={JobStatus.RUNNING} totalTraces={10} />
      </MemoryRouter>,
    );

    expect(screen.getByText('Cancel')).toBeInTheDocument();
  });

  test('does not offer Cancel once the job is complete', () => {
    renderWithIntl(
      <MemoryRouter>
        <IssueDetectionProgress jobStatus={JobStatus.SUCCEEDED} totalTraces={10} result={{ issues: 3 }} />
      </MemoryRouter>,
    );

    expect(screen.queryByText('Cancel')).not.toBeInTheDocument();
  });

  test('asks for confirmation before canceling the run', async () => {
    const mockCancelAsync = jest.fn<(_: { jobId: string; runUuid: string }) => Promise<void>>(() => Promise.resolve());
    jest.mocked(useCancelJob).mockReturnValue({
      cancelJob: jest.fn(),
      cancelJobAsync: mockCancelAsync,
      isCancelling: false,
    } as any);

    renderWithIntl(
      <MemoryRouter>
        <IssueDetectionProgress jobStatus={JobStatus.RUNNING} jobId="job-1" totalTraces={10} />
      </MemoryRouter>,
    );

    await userEvent.click(screen.getByText('Cancel'));

    // Confirmation dialog appears; nothing is canceled yet
    expect(screen.getByText('Cancel issue detection?')).toBeInTheDocument();
    expect(mockCancelAsync).not.toHaveBeenCalled();

    await userEvent.click(screen.getByText('Cancel run'));
    await waitFor(() => {
      expect(mockCancelAsync).toHaveBeenCalledWith({ jobId: 'job-1', runUuid: 'test-run-uuid' });
    });
  });
});

describe('IssueDetectionProgress telemetry', () => {
  let mockLogTelemetryEvent: jest.Mock;

  beforeEach(() => {
    mockLogTelemetryEvent = jest.fn();
    useLogTelemetryEvent.mockReturnValue(mockLogTelemetryEvent);
  });

  test('logs telemetry event when job succeeds', async () => {
    await act(async () => {
      renderWithIntl(
        <MemoryRouter>
          <IssueDetectionProgress
            jobStatus={JobStatus.SUCCEEDED}
            totalTraces={10}
            result={{ issues: 3, total_cost_usd: 0.05 }}
          />
        </MemoryRouter>,
      );
    });

    expect(mockLogTelemetryEvent).toHaveBeenCalledTimes(1);
    expect(mockLogTelemetryEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        componentId: 'mlflow.issue-detection.completed',
        componentViewId: 'test-run-uuid',
        value: JSON.stringify({ totalTraces: 10, identifiedIssues: 3, totalCostUsd: 0.05 }),
      }),
    );
  });

  test('does not log telemetry when job has not succeeded', async () => {
    await act(async () => {
      renderWithIntl(
        <MemoryRouter>
          <IssueDetectionProgress jobStatus={JobStatus.RUNNING} totalTraces={10} />
        </MemoryRouter>,
      );
    });

    expect(mockLogTelemetryEvent).not.toHaveBeenCalled();
  });
});

describe('IssueDetectionProgress low-result callout', () => {
  const { useSearchIssuesQuery } = jest.requireMock('../hooks/useSearchIssuesQuery') as {
    useSearchIssuesQuery: jest.Mock;
  };

  const renderProgress = async (jobStatus: JobStatus, issues: unknown[], tracesAnalyzed = 5) => {
    useSearchIssuesQuery.mockReturnValue({ issues });
    await act(async () => {
      renderWithIntl(
        <MemoryRouter>
          <IssueDetectionProgress
            jobStatus={jobStatus}
            totalTraces={tracesAnalyzed}
            result={{ issues: issues.length, total_traces_analyzed: tracesAnalyzed }}
          />
        </MemoryRouter>,
      );
    });
  };

  test('shows guidance when a succeeded job found no issues', async () => {
    await renderProgress(JobStatus.SUCCEEDED, []);

    expect(screen.getByText("0 issues doesn't always mean all clear")).toBeInTheDocument();
    expect(screen.getByText(/Only 5 traces were analyzed/)).toBeInTheDocument();
    expect(screen.getByTestId('low-results-run-again')).toBeInTheDocument();
    expect(screen.getByText('Add user feedback')).toBeInTheDocument();
    expect(screen.getByText('Annotate traces')).toBeInTheDocument();
  });

  test('shows single-issue guidance variant', async () => {
    await renderProgress(JobStatus.SUCCEEDED, [{ issue_id: 'iss-1' }]);

    expect(screen.getByText('Only 1 issue found. There may be more.')).toBeInTheDocument();
  });

  test('omits the add-more-traces suggestion for large runs', async () => {
    await renderProgress(JobStatus.SUCCEEDED, [], 100);

    expect(screen.getByText("0 issues doesn't always mean all clear")).toBeInTheDocument();
    expect(screen.queryByText(/traces were analyzed/)).not.toBeInTheDocument();
  });

  test('does not show guidance when multiple issues are found', async () => {
    await renderProgress(JobStatus.SUCCEEDED, [{ issue_id: 'iss-1' }, { issue_id: 'iss-2' }]);

    expect(screen.queryByText("0 issues doesn't always mean all clear")).not.toBeInTheDocument();
    expect(screen.queryByText('Only 1 issue found. There may be more.')).not.toBeInTheDocument();
  });

  test('does not show guidance while the job is running', async () => {
    await renderProgress(JobStatus.RUNNING, []);

    expect(screen.queryByText("0 issues doesn't always mean all clear")).not.toBeInTheDocument();
  });
});
