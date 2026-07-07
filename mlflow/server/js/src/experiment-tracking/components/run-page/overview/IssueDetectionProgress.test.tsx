import { jest, describe, test, expect, beforeEach } from '@jest/globals';
import { act, renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { IssueDetectionProgress } from './IssueDetectionProgress';
import { JobStatus } from '../hooks/useFetchJobStatus';

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
  useCancelJob: jest.fn(() => ({ cancelJob: jest.fn(), isCancelling: false })),
}));

const { useLogTelemetryEvent } = jest.requireMock('@mlflow/mlflow/src/telemetry/hooks/useLogTelemetryEvent') as {
  useLogTelemetryEvent: jest.Mock;
};

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
