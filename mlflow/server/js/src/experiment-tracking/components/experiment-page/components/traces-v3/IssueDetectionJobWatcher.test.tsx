import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionJobWatcher } from './IssueDetectionJobWatcher';
import { JobStatus, useFetchJobStatus } from '../../../run-page/hooks/useFetchJobStatus';
import { useActiveIssueDetectionRun } from './hooks/useActiveIssueDetectionRun';
import { useNavigate } from '../../../../../common/utils/RoutingUtils';

jest.mock('./hooks/useActiveIssueDetectionRun', () => ({
  useActiveIssueDetectionRun: jest.fn(),
}));
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
}));

const mockJobStatus = (status?: JobStatus, stage?: string, result?: unknown) => {
  jest.mocked(useFetchJobStatus).mockReturnValue({
    status,
    result,
    status_details: stage ? { stage } : undefined,
    isLoading: false,
    isFetching: false,
    refetch: jest.fn(),
    error: null,
  });
};

describe('IssueDetectionJobWatcher', () => {
  let mockNavigate: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    mockNavigate = jest.fn();
    jest.mocked(useNavigate).mockReturnValue(mockNavigate);
    jest.mocked(useActiveIssueDetectionRun).mockReturnValue({ activeRun: undefined });
    mockJobStatus(undefined);
  });

  test('renders no visible UI', () => {
    mockJobStatus(JobStatus.RUNNING);
    const { container } = renderWithDesignSystem(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 5 }}
      />,
    );

    expect(container.querySelectorAll('button').length).toBe(0);
  });

  test('shows started notification when a job is submitted from this session', async () => {
    mockJobStatus(JobStatus.PENDING);

    renderWithDesignSystem(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 48 }}
      />,
    );

    expect(await screen.findByText('Issue detection started')).toBeInTheDocument();
    expect(screen.getByText(/Analyzing 48 traces/)).toBeInTheDocument();
    expect(screen.getByText('View progress')).toBeInTheDocument();
  });

  test('shows completion notification with issues link when job succeeds', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 48 }}
      />,
    );
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.SUCCEEDED, undefined, { issues: 4, total_traces_analyzed: 48 });
    rerender(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 48 }}
      />,
    );

    expect(await screen.findByText('Issue detection completed')).toBeInTheDocument();
    expect(screen.getByText('Found 4 issues across 48 traces.')).toBeInTheDocument();

    await userEvent.click(screen.getByText('View issues'));
    expect(mockNavigate).toHaveBeenCalledWith(expect.stringContaining('/experiments/exp-1/evaluation-runs/run-1'));
  });

  test('notifies for a running job discovered in the experiment once it completes', async () => {
    jest.mocked(useActiveIssueDetectionRun).mockReturnValue({ activeRun: { runId: 'run-1', jobId: 'job-1' } });
    mockJobStatus(JobStatus.RUNNING);

    const { rerender } = renderWithDesignSystem(<IssueDetectionJobWatcher experimentId="exp-1" />);

    mockJobStatus(JobStatus.SUCCEEDED, undefined, { issues: 2, total_traces_analyzed: 10 });
    rerender(<IssueDetectionJobWatcher experimentId="exp-1" />);

    expect(await screen.findByText('Issue detection completed')).toBeInTheDocument();
    expect(screen.getByText('Found 2 issues across 10 traces.')).toBeInTheDocument();
  });

  test('low-result completion links to details instead of issues', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 5 }}
      />,
    );
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.SUCCEEDED, undefined, { issues: 0, total_traces_analyzed: 5 });
    rerender(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 5 }}
      />,
    );

    expect(await screen.findByText('Found no issues across 5 traces.')).toBeInTheDocument();
    expect(screen.getByText('View details')).toBeInTheDocument();
    expect(screen.queryByText('View issues')).not.toBeInTheDocument();
  });

  test('shows failure notification when job fails', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 5 }}
      />,
    );
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.FAILED, undefined, 'boom');
    rerender(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 5 }}
      />,
    );

    expect(await screen.findByText('Issue detection failed')).toBeInTheDocument();
  });

  test('does not show any notification for canceled jobs', async () => {
    mockJobStatus(JobStatus.RUNNING);
    const { rerender } = renderWithDesignSystem(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 5 }}
      />,
    );
    await screen.findByText('Issue detection started');

    mockJobStatus(JobStatus.CANCELED);
    rerender(
      <IssueDetectionJobWatcher
        experimentId="exp-1"
        submittedJob={{ jobId: 'job-1', runId: 'run-1', traceCount: 5 }}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText('Issue detection completed')).not.toBeInTheDocument();
    });
    expect(screen.queryByText('Issue detection failed')).not.toBeInTheDocument();
  });
});
