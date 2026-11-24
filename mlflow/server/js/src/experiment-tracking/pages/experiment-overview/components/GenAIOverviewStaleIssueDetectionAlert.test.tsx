import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../../common/utils/TestUtils.react18';
import { GenAIOverviewStaleIssueDetectionAlert } from './GenAIOverviewStaleIssueDetectionAlert';

const readyState = {
  status: 'ready' as const,
  latestRunUuid: 'run-1',
  elapsedDays: 9,
  newTraceCount: 12,
};

describe('GenAIOverviewStaleIssueDetectionAlert', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  test('starts detection and reflects an active detection', async () => {
    const onDetectNewIssues = jest.fn();
    const { rerender } = renderWithDesignSystem(
      <GenAIOverviewStaleIssueDetectionAlert
        experimentId="experiment-1"
        state={readyState}
        issueDetectionPhase="idle"
        canStartDetection
        onDetectNewIssues={onDetectNewIssues}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent(
      "It's been 9 days and 12 new traces since your last issue detection.",
    );
    await userEvent.click(screen.getByRole('button', { name: 'Detect new issues' }));
    expect(onDetectNewIssues).toHaveBeenCalledTimes(1);

    rerender(
      <GenAIOverviewStaleIssueDetectionAlert
        experimentId="experiment-1"
        state={readyState}
        issueDetectionPhase="running"
        canStartDetection={false}
        onDetectNewIssues={onDetectNewIssues}
      />,
    );
    expect(screen.getByRole('button', { name: 'Detecting issues...' })).toBeDisabled();
  });

  test('dismisses only the reminder for the current completed run', async () => {
    const { rerender } = renderWithDesignSystem(
      <GenAIOverviewStaleIssueDetectionAlert
        experimentId="experiment-1"
        state={readyState}
        issueDetectionPhase="idle"
        canStartDetection
        onDetectNewIssues={jest.fn()}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Dismiss stale issue detection reminder' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    rerender(
      <GenAIOverviewStaleIssueDetectionAlert
        experimentId="experiment-1"
        state={{ ...readyState, latestRunUuid: 'run-2' }}
        issueDetectionPhase="idle"
        canStartDetection
        onDetectNewIssues={jest.fn()}
      />,
    );
    expect(screen.getByRole('alert')).toBeVisible();
  });
});
