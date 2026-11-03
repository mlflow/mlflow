import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '../../../common/utils/TestUtils.react18';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { generatePath, useLocation } from '../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../routes';
import ExperimentGenAIJourneyOverviewPage from './ExperimentGenAIJourneyOverviewPage';

const mockOpenAssistantPanel = jest.fn();
const mockPrefillAssistantPrompt = jest.fn();
const mockSendAssistantMessage = jest.fn();
const mockUseAssistant = jest.fn();
const mockShouldEnableIssueDetection = jest.fn();
const mockShouldEnableExperimentIssuesTab = jest.fn();
const mockUseDetectionTraceIds = jest.fn();
const mockUseGenAIOverviewTraceState = jest.fn();
const mockUseGenAIOverviewAnalyzeState = jest.fn();
const mockUseGenAIOverviewEvalState = jest.fn();
const mockUseGenAIOverviewMonitorState = jest.fn();
const mockUseGenAIOverviewStaleIssueDetectionState = jest.fn();
const mockUseExperimentContainsTraces = jest.fn();
const mockUseGenAIOverviewIssueDetectionProgress = jest.fn();

const CurrentTestLocation = () => {
  const { pathname, search } = useLocation();
  return (
    <div role="status" aria-label="Current test location">
      {`${pathname}${search}`}
    </div>
  );
};

jest.mock('../../../assistant', () => ({
  useAssistant: () => mockUseAssistant(),
}));

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableIssueDetection: () => mockShouldEnableIssueDetection(),
  shouldEnableExperimentIssuesTab: () => mockShouldEnableExperimentIssuesTab(),
}));

jest.mock('../experiment-issues/useIssueDetection', () => ({
  useDetectionTraceIds: (...args: unknown[]) => mockUseDetectionTraceIds(...args),
}));

jest.mock('./hooks/useGenAIOverviewState', () => ({
  useGenAIOverviewTraceState: (...args: unknown[]) => mockUseGenAIOverviewTraceState(...args),
}));

jest.mock('./hooks/useGenAIOverviewAnalyzeState', () => ({
  useGenAIOverviewAnalyzeState: (...args: unknown[]) => mockUseGenAIOverviewAnalyzeState(...args),
}));

jest.mock('./hooks/useGenAIOverviewEvalState', () => ({
  useGenAIOverviewEvalState: (...args: unknown[]) => mockUseGenAIOverviewEvalState(...args),
}));

jest.mock('./hooks/useGenAIOverviewMonitorState', () => ({
  useGenAIOverviewMonitorState: (...args: unknown[]) => mockUseGenAIOverviewMonitorState(...args),
}));

jest.mock('./hooks/useGenAIOverviewStaleIssueDetectionState', () => ({
  useGenAIOverviewStaleIssueDetectionState: (...args: unknown[]) =>
    mockUseGenAIOverviewStaleIssueDetectionState(...args),
}));

jest.mock('./hooks/useGenAIOverviewIssueDetectionProgress', () => ({
  useGenAIOverviewIssueDetectionProgress: (...args: unknown[]) => mockUseGenAIOverviewIssueDetectionProgress(...args),
}));

jest.mock('../../components/traces/hooks/useExperimentContainsTraces', () => ({
  useExperimentContainsTraces: (...args: unknown[]) => mockUseExperimentContainsTraces(...args),
}));

jest.mock('./components/GenAIOverviewTraceActivityChart', () => ({
  GenAIOverviewTraceActivityChart: ({ onAskAssistant }: { onAskAssistant?: (context: unknown) => void }) =>
    onAskAssistant ? (
      <button
        onClick={() =>
          onAskAssistant({
            stage: 'trace',
            label: 'Trace requests · Aug 1–8',
            startTimeMs: Date.UTC(2026, 7, 1),
            endTimeMsExclusive: Date.UTC(2026, 7, 8),
            traceCount: 12,
          })
        }
      >
        Attach trace chart
      </button>
    ) : null,
}));

jest.mock('../../components/experiment-page/components/traces-v3/IssueDetectionModal', () => ({
  IssueDetectionModal: ({ availableTraceIds }: { availableTraceIds: string[] }) => (
    <div role="dialog" aria-label="Detect Issues">
      <span>{availableTraceIds.join(',')}</span>
    </div>
  ),
}));

describe('ExperimentGenAIJourneyOverviewPage', () => {
  const renderPage = () =>
    renderWithDesignSystem(
      <TestRouter
        routes={[
          testRoute(<ExperimentGenAIJourneyOverviewPage />, RoutePaths.experimentPageTabJourneyOverview),
          testRoute(<CurrentTestLocation />, RoutePaths.experimentPageTabTraces),
          testRoute(<CurrentTestLocation />, RoutePaths.experimentPageTabIssues),
        ]}
        initialEntries={[
          generatePath(RoutePaths.experimentPageTabJourneyOverview, {
            experimentId: '123',
          }),
        ]}
      />,
    );

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    mockUseAssistant.mockReturnValue({
      canUseAssistant: true,
      openPanel: mockOpenAssistantPanel,
      prefillPrompt: mockPrefillAssistantPrompt,
      sendMessageWhenReady: mockSendAssistantMessage,
    });
    mockShouldEnableIssueDetection.mockReturnValue(false);
    mockShouldEnableExperimentIssuesTab.mockReturnValue(false);
    mockUseDetectionTraceIds.mockReturnValue({
      data: ['trace-1', 'trace-2'],
      isFetching: false,
      error: null,
    });
    mockUseGenAIOverviewTraceState.mockReturnValue({
      state: {
        status: 'ready',
        totalCount: 12,
        activity: [{ timestampMs: Date.UTC(2026, 7, 1), count: 12 }],
      },
      retry: jest.fn(),
    });
    mockUseGenAIOverviewAnalyzeState.mockReturnValue({ state: { status: 'never-run' }, refetch: jest.fn() });
    mockUseGenAIOverviewEvalState.mockReturnValue({ status: 'empty' });
    mockUseGenAIOverviewMonitorState.mockReturnValue({ status: 'empty' });
    mockUseGenAIOverviewStaleIssueDetectionState.mockReturnValue({ status: 'hidden' });
    mockUseExperimentContainsTraces.mockReturnValue({ containsTraces: true, isLoading: false });
    mockUseGenAIOverviewIssueDetectionProgress.mockReturnValue({
      phase: 'idle',
      canStartDetection: true,
      isCancelling: false,
      cancelError: null,
    });
  });

  test('submits the selected chart through the existing Assistant', async () => {
    renderPage();

    expect(await screen.findByRole('heading', { name: "Let's improve your agent with MLflow" })).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Attach trace chart' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Ask Assistant about the selected chart' }), 'Why?');
    await userEvent.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(mockSendAssistantMessage).toHaveBeenCalledTimes(1));
    expect(mockOpenAssistantPanel).toHaveBeenCalledTimes(1);
    expect(mockSendAssistantMessage).toHaveBeenCalledWith(expect.stringContaining('Why?'));
    expect(mockSendAssistantMessage).toHaveBeenCalledWith(
      expect.stringContaining('## Selected Overview chart context'),
    );
    expect(mockSendAssistantMessage).toHaveBeenCalledWith(expect.stringContaining('"traceCount": 12'));
  });

  test('uses the 30-day range for trace activity and trace navigation', async () => {
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'View traces' }));

    expect(mockUseGenAIOverviewTraceState).toHaveBeenLastCalledWith('123', false);
    expect(mockUseGenAIOverviewAnalyzeState).toHaveBeenLastCalledWith('123', false);
    expect(await screen.findByRole('status', { name: 'Current test location' })).toHaveTextContent(
      '/experiments/123/traces?startTimeLabel=LAST_30_DAYS',
    );
  });

  test('hides Assistant actions when the Assistant is unavailable', async () => {
    mockUseAssistant.mockReturnValue({
      canUseAssistant: false,
      openPanel: mockOpenAssistantPanel,
      prefillPrompt: mockPrefillAssistantPrompt,
      sendMessageWhenReady: mockSendAssistantMessage,
    });

    renderPage();

    expect(await screen.findByRole('heading', { name: "Let's improve your agent with MLflow" })).toBeVisible();
    expect(screen.queryByRole('textbox', { name: 'Ask MLflow about this agent' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Attach trace chart' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Detect issues' })).not.toBeInTheDocument();
  });

  test('explains how to enable issue detection and links to its documentation', async () => {
    renderPage();

    expect(
      await screen.findByText(
        'Issue detection is not enabled for this MLflow server. Ask an administrator to restart the server with MLFLOW_ENABLE_AI_GATEWAY=true.',
      ),
    ).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Detect issues' })).not.toBeInTheDocument();
    const docsLink = screen.getByRole('link', { name: 'Learn more about enabling issue detection' });
    expect(docsLink).toHaveTextContent('Learn more');
    expect(docsLink).toHaveAttribute(
      'href',
      'https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/',
    );
    expect(docsLink).toHaveAttribute('target', '_blank');
    expect(docsLink).toHaveAttribute('rel', 'noopener noreferrer');
    expect(mockOpenAssistantPanel).not.toHaveBeenCalled();
    expect(mockPrefillAssistantPrompt).not.toHaveBeenCalled();
  });

  test('opens native issue detection only after its traces are loaded', async () => {
    mockShouldEnableIssueDetection.mockReturnValue(true);
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'Detect issues' }));
    expect(await screen.findByRole('dialog', { name: 'Detect Issues' })).toHaveTextContent('trace-1,trace-2');
    expect(mockUseDetectionTraceIds).toHaveBeenLastCalledWith('123', true);
  });

  test('does not mount issue detection with a false empty state while traces are loading', async () => {
    mockShouldEnableIssueDetection.mockReturnValue(true);
    mockUseDetectionTraceIds.mockReturnValue({ data: undefined, isFetching: true, error: null });
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'Detect issues' }));

    expect(screen.queryByRole('dialog', { name: 'Detect Issues' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'loading' })).toHaveAttribute('loading', 'true');
  });

  test('keeps completed results when issue detection is unavailable', async () => {
    mockUseGenAIOverviewAnalyzeState.mockReturnValue({
      state: {
        status: 'completed',
        runUuid: 'run-1',
        latestRunCompletedAtMs: 1,
        issueCount: 2,
        issuesCreatedInLastSevenDays: 0,
        activity: [],
      },
      refetch: jest.fn(),
    });

    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'View latest results' }));
    expect(await screen.findByRole('status', { name: 'Current test location' })).toHaveTextContent(
      '/experiments/123/issues',
    );
    expect(screen.queryByText(/MLFLOW_ENABLE_AI_GATEWAY=true/)).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Learn more about enabling issue detection' })).not.toBeInTheDocument();
  });

  test('links View issues to the experiment Issues tab when it is enabled', async () => {
    mockShouldEnableExperimentIssuesTab.mockReturnValue(true);
    mockUseGenAIOverviewAnalyzeState.mockReturnValue({
      state: {
        status: 'completed',
        runUuid: 'run-1',
        latestRunCompletedAtMs: 1,
        issueCount: 2,
        issuesCreatedInLastSevenDays: 0,
        activity: [],
      },
      refetch: jest.fn(),
    });

    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'View issues' }));
    expect(await screen.findByRole('status', { name: 'Current test location' })).toHaveTextContent(
      '/experiments/123/issues',
    );
  });

  test('shows cancellation errors instead of leaving a rejected cancellation silent', async () => {
    mockUseGenAIOverviewIssueDetectionProgress.mockReturnValue({
      phase: 'running',
      canStartDetection: false,
      cancelDetection: jest.fn(),
      isCancelling: false,
      cancelError: new Error('cancel failed'),
    });

    renderPage();

    expect(await screen.findByRole('alert')).toHaveTextContent('Unable to cancel issue detection');
    expect(screen.getByRole('alert')).toHaveTextContent('cancel failed');
  });

  test('does not show a waiting status before the first trace arrives', async () => {
    mockUseGenAIOverviewTraceState.mockReturnValue({
      state: { status: 'empty', activity: [] },
      retry: jest.fn(),
    });
    mockUseExperimentContainsTraces.mockReturnValue({ containsTraces: false, isLoading: false });

    renderPage();

    expect(
      await screen.findByRole('heading', { name: 'Send your first trace to MLflow in just a minute' }),
    ).toBeVisible();
    expect(screen.queryByText('Waiting for traces…')).not.toBeInTheDocument();
  });
});
