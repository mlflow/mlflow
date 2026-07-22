import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { act, renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import {
  clearSubmittedIssueDetectionJob,
  getSubmittedIssueDetectionJob,
  getSubmittedIssueDetectionJobs,
  IssueDetectionJobNotifications,
  recordSubmittedIssueDetectionJob,
  type SubmittedIssueDetectionJob,
} from './IssueDetectionJobNotifications';
import { JobStatus, useFetchJobStatus, type UseFetchJobStatusResult } from '../../../run-page/hooks/useFetchJobStatus';
import { useNavigate, useSearchParams } from '../../../../../common/utils/RoutingUtils';

jest.mock('../../../run-page/hooks/useFetchJobStatus', () => ({
  ...jest.requireActual<typeof import('../../../run-page/hooks/useFetchJobStatus')>(
    '../../../run-page/hooks/useFetchJobStatus',
  ),
  useFetchJobStatus: jest.fn(),
}));
jest.mock('../../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/RoutingUtils')>(
    '../../../../../common/utils/RoutingUtils',
  ),
  useNavigate: jest.fn(),
  useSearchParams: jest.fn(),
}));

const submittedJob: SubmittedIssueDetectionJob = {
  experimentId: 'exp-1',
  jobId: 'job-1',
  runId: 'run-1',
  traceCount: 48,
};

const secondSubmittedJob: SubmittedIssueDetectionJob = {
  experimentId: 'exp-1',
  jobId: 'job-2',
  runId: 'run-2',
  traceCount: 12,
};

const createMockJobStatus = (status?: JobStatus, stage?: string, result?: unknown): UseFetchJobStatusResult => ({
  status,
  result,
  status_details: stage ? { stage } : undefined,
  isLoading: false,
  isFetching: false,
  refetch: jest.fn(),
  error: null,
});

let mockJobStatuses: Record<string, UseFetchJobStatusResult>;

const mockJobStatus = (status?: JobStatus, stage?: string, result?: unknown, jobId = submittedJob.jobId) => {
  mockJobStatuses[jobId] = createMockJobStatus(status, stage, result);
  jest
    .mocked(useFetchJobStatus)
    .mockImplementation(({ jobId }) =>
      jobId ? (mockJobStatuses[jobId] ?? createMockJobStatus(undefined)) : createMockJobStatus(undefined),
    );
};

describe('IssueDetectionJobNotifications', () => {
  let mockNavigate: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    mockJobStatuses = {};
    clearSubmittedIssueDetectionJob();
    mockNavigate = jest.fn();
    jest.mocked(useNavigate).mockReturnValue(mockNavigate);
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), jest.fn()] as any);
    jest.mocked(useFetchJobStatus).mockImplementation(() => createMockJobStatus(undefined));
  });

  test('renders no visible UI', () => {
    mockJobStatus(JobStatus.RUNNING);
    const { container } = renderWithDesignSystem(<IssueDetectionJobNotifications />);

    expect(container.querySelectorAll('button').length).toBe(0);
  });

  test('shows started notification when a job is submitted from this session', async () => {
    mockJobStatus(JobStatus.PENDING);

    renderWithDesignSystem(<IssueDetectionJobNotifications />);

    act(() => {
      recordSubmittedIssueDetectionJob(submittedJob);
    });

    expect(await screen.findByText('Issue detection started')).toBeInTheDocument();
    expect(screen.getByText(/Analyzing 48 traces/)).toBeInTheDocument();
    expect(screen.getByText('View progress')).toBeInTheDocument();
  });

  test('preserves experiment query params in notification links', async () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([
        new URLSearchParams('selectedTraceId=trace-123&startTimeLabel=LAST_24_HOURS&workspace=default'),
        jest.fn(),
      ] as any);
    mockJobStatus(JobStatus.PENDING);

    renderWithDesignSystem(<IssueDetectionJobNotifications />);

    act(() => {
      recordSubmittedIssueDetectionJob(submittedJob);
    });

    await userEvent.click(await screen.findByText('View progress'));

    const [route] = mockNavigate.mock.calls[0];
    expect(route).toContain('?startTimeLabel=LAST_24_HOURS&workspace=default');
    expect(route).not.toContain('selectedTraceId');
  });

  test('shows completion notification with issues link when job succeeds', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(<IssueDetectionJobNotifications />);

    act(() => {
      recordSubmittedIssueDetectionJob(submittedJob);
    });
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.SUCCEEDED, undefined, { issues: 4, total_traces_analyzed: 48 });
    rerender(<IssueDetectionJobNotifications />);

    expect(await screen.findByText('Issue detection completed')).toBeInTheDocument();
    expect(screen.getByText('Found 4 issues across 48 traces.')).toBeInTheDocument();

    await userEvent.click(screen.getByText('View issues'));
    expect(mockNavigate).toHaveBeenCalledWith(expect.stringContaining('/experiments/exp-1/evaluation-runs/run-1'));
  });

  test('tracks concurrent submitted jobs from the current session', async () => {
    mockJobStatus(JobStatus.RUNNING, undefined, undefined, submittedJob.jobId);
    mockJobStatus(JobStatus.RUNNING, undefined, undefined, secondSubmittedJob.jobId);

    renderWithDesignSystem(<IssueDetectionJobNotifications />);

    act(() => {
      recordSubmittedIssueDetectionJob(submittedJob);
      recordSubmittedIssueDetectionJob(secondSubmittedJob);
    });

    await waitFor(() => {
      expect(screen.getAllByText('Issue detection started')).toHaveLength(2);
    });
    expect(screen.getByText(/Analyzing 48 traces/)).toBeInTheDocument();
    expect(screen.getByText(/Analyzing 12 traces/)).toBeInTheDocument();
    await waitFor(() => {
      expect(useFetchJobStatus).toHaveBeenCalledWith({ jobId: 'job-1', enabled: true });
      expect(useFetchJobStatus).toHaveBeenCalledWith({ jobId: 'job-2', enabled: true });
    });
    expect(getSubmittedIssueDetectionJobs()).toEqual([submittedJob, secondSubmittedJob]);
  });

  test('recovers a submitted job from session storage after reload', async () => {
    recordSubmittedIssueDetectionJob(submittedJob);
    mockJobStatus(JobStatus.RUNNING);

    const { rerender } = renderWithDesignSystem(<IssueDetectionJobNotifications />);

    expect(screen.queryByText('Issue detection started')).not.toBeInTheDocument();

    mockJobStatus(JobStatus.SUCCEEDED, undefined, { issues: 2, total_traces_analyzed: 10 });
    rerender(<IssueDetectionJobNotifications />);

    expect(await screen.findByText('Issue detection completed')).toBeInTheDocument();
    expect(screen.getByText('Found 2 issues across 10 traces.')).toBeInTheDocument();
  });

  test('removes only the completed job from session storage', async () => {
    recordSubmittedIssueDetectionJob(submittedJob);
    recordSubmittedIssueDetectionJob(secondSubmittedJob);
    mockJobStatus(JobStatus.RUNNING, undefined, undefined, submittedJob.jobId);
    mockJobStatus(JobStatus.SUCCEEDED, undefined, { issues: 3, total_traces_analyzed: 12 }, secondSubmittedJob.jobId);

    renderWithDesignSystem(<IssueDetectionJobNotifications />);

    expect(await screen.findByText('Issue detection completed')).toBeInTheDocument();
    await waitFor(() => {
      expect(getSubmittedIssueDetectionJobs()).toEqual([submittedJob]);
    });
  });

  test('low-result completion links to details instead of issues', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(<IssueDetectionJobNotifications />);

    act(() => {
      recordSubmittedIssueDetectionJob({ ...submittedJob, traceCount: 5 });
    });
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.SUCCEEDED, undefined, { issues: 0, total_traces_analyzed: 5 });
    rerender(<IssueDetectionJobNotifications />);

    expect(await screen.findByText('Found no issues across 5 traces.')).toBeInTheDocument();
    expect(screen.getByText('View details')).toBeInTheDocument();
    expect(screen.queryByText('View issues')).not.toBeInTheDocument();
  });

  test('shows failure notification when job fails', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(<IssueDetectionJobNotifications />);

    act(() => {
      recordSubmittedIssueDetectionJob({ ...submittedJob, traceCount: 5 });
    });
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.FAILED, undefined, 'boom');
    rerender(<IssueDetectionJobNotifications />);

    expect(await screen.findByText('Issue detection failed')).toBeInTheDocument();
  });

  test('does not show any notification for canceled jobs', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(<IssueDetectionJobNotifications />);

    act(() => {
      recordSubmittedIssueDetectionJob({ ...submittedJob, traceCount: 5 });
    });
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.CANCELED);
    rerender(<IssueDetectionJobNotifications />);

    await waitFor(() => {
      expect(getSubmittedIssueDetectionJob()).toBeNull();
    });
    expect(screen.queryByText('Issue detection completed')).not.toBeInTheDocument();
    expect(screen.queryByText('Issue detection failed')).not.toBeInTheDocument();
  });
});
