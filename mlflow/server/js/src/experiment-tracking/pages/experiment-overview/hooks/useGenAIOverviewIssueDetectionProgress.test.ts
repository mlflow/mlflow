import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { JobStatus } from '../../../components/run-page/hooks/useFetchJobStatus';
import { useGenAIOverviewIssueDetectionProgress } from './useGenAIOverviewIssueDetectionProgress';

const mockUseLatestIssueDetection = jest.fn();
const mockCancelJobAsync = jest.fn<(_: { jobId: string; runUuid: string }) => Promise<void>>(() => Promise.resolve());
const mockUseCancelJob = jest.fn();

jest.mock('../../experiment-issues/useIssueDetection', () => ({
  useLatestIssueDetection: (...args: unknown[]) => mockUseLatestIssueDetection(...args),
}));

jest.mock('../../../components/run-page/hooks/useCancelJob', () => ({
  useCancelJob: () => mockUseCancelJob(),
}));

describe('useGenAIOverviewIssueDetectionProgress', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseLatestIssueDetection.mockReturnValue({ run: undefined, jobId: undefined, status: undefined });
    mockUseCancelJob.mockReturnValue({
      cancelJobAsync: mockCancelJobAsync,
      isCancelling: false,
      error: null,
    });
  });

  test.each([
    [undefined, 'starting'],
    [JobStatus.PENDING, 'starting'],
    [JobStatus.RUNNING, 'running'],
  ])('maps active status %s to the %s phase', (status, expectedPhase) => {
    mockUseLatestIssueDetection.mockReturnValue({
      run: { info: { run_id: 'run-1' } },
      jobId: 'job-1',
      status,
    });

    const { result } = renderHook(() => useGenAIOverviewIssueDetectionProgress('experiment-1'));

    expect(result.current.phase).toBe(expectedPhase);
    expect(result.current.canStartDetection).toBe(false);
  });

  test('allows a new detection after the shared state reaches a terminal status', () => {
    mockUseLatestIssueDetection.mockReturnValue({
      run: { info: { run_id: 'run-1' } },
      jobId: 'job-1',
      status: JobStatus.FAILED,
    });

    const { result } = renderHook(() => useGenAIOverviewIssueDetectionProgress('experiment-1'));

    expect(result.current.phase).toBe('idle');
    expect(result.current.canStartDetection).toBe(true);
  });

  test('cancels the shared job and exposes cancellation failures', async () => {
    const cancelError = new Error('cancel failed');
    mockUseLatestIssueDetection.mockReturnValue({
      run: { info: { run_id: 'run-1' } },
      jobId: 'job-1',
      status: JobStatus.RUNNING,
    });
    mockUseCancelJob.mockReturnValue({
      cancelJobAsync: mockCancelJobAsync,
      isCancelling: false,
      error: cancelError,
    });

    const { result } = renderHook(() => useGenAIOverviewIssueDetectionProgress('experiment-1'));
    await act(async () => result.current.cancelDetection?.());

    expect(mockCancelJobAsync).toHaveBeenCalledWith({ jobId: 'job-1', runUuid: 'run-1' });
    expect(result.current.cancelError).toBe(cancelError);
  });
});
