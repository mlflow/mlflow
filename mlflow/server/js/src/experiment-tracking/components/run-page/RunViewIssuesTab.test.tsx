import { jest, describe, test, expect, beforeEach } from '@jest/globals';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { RunViewIssuesTab } from './RunViewIssuesTab';
import { JobStatus, useFetchJobStatus, type UseFetchJobStatusResult } from './hooks/useFetchJobStatus';
import { useSearchIssuesQuery } from './hooks/useSearchIssuesQuery';

jest.mock('./hooks/useSearchIssuesQuery', () => ({
  useSearchIssuesQuery: jest.fn(),
}));

jest.mock('./hooks/useFetchJobStatus', () => ({
  ...jest.requireActual<typeof import('./hooks/useFetchJobStatus')>('./hooks/useFetchJobStatus'),
  useFetchJobStatus: jest.fn(),
}));

const mockUseSearchIssuesQuery = jest.mocked(useSearchIssuesQuery);
const mockUseFetchJobStatus = jest.mocked(useFetchJobStatus);

const mockJobStatus = (status?: JobStatus, error: Error | null = null) => {
  mockUseFetchJobStatus.mockReturnValue({
    status,
    result: undefined,
    status_details: undefined,
    isLoading: false,
    isFetching: false,
    refetch: jest.fn(),
    error,
  } satisfies UseFetchJobStatusResult);
};

describe('RunViewIssuesTab', () => {
  const renderTab = (jobId?: string) =>
    renderWithIntl(
      <MemoryRouter>
        <RunViewIssuesTab experimentId="exp-1" runUuid="run-1" jobId={jobId} />
      </MemoryRouter>,
    );

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseSearchIssuesQuery.mockReturnValue({ issues: [], isLoading: false } as any);
    mockJobStatus();
  });

  test('hides low-result guidance while issue detection is still running', () => {
    mockJobStatus(JobStatus.RUNNING);

    renderTab('job-1');

    expect(screen.getByText('No issues found')).toBeInTheDocument();
    expect(screen.queryByText("0 issues doesn't always mean all clear")).not.toBeInTheDocument();
  });

  test('shows low-result guidance after issue detection completes', () => {
    mockJobStatus(JobStatus.SUCCEEDED);

    renderTab('job-1');

    expect(screen.getByText("0 issues doesn't always mean all clear")).toBeInTheDocument();
  });

  test('shows low-result guidance for legacy runs without a job id', () => {
    renderTab();

    expect(screen.getByText("0 issues doesn't always mean all clear")).toBeInTheDocument();
  });
});
