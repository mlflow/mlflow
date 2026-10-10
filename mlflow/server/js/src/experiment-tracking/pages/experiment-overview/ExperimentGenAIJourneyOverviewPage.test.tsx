import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '../../../common/utils/TestUtils.react18';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { generatePath } from '../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../routes';
import ExperimentGenAIJourneyOverviewPage from './ExperimentGenAIJourneyOverviewPage';

const mockOpenAssistantPanel = jest.fn();
const mockPrefillAssistantPrompt = jest.fn();
const mockSendAssistantMessage = jest.fn();
const mockUseAssistant = jest.fn();
const mockShouldEnableIssueDetection = jest.fn();
const mockUseSearchMlflowTraces = jest.fn();
const mockUseGenAIOverviewTraceState = jest.fn();
const mockUseGenAIOverviewAnalyzeState = jest.fn();
const mockUseGenAIOverviewEvalState = jest.fn();
const mockUseGenAIOverviewMonitorState = jest.fn();
const mockUseGenAIOverviewStaleIssueDetectionState = jest.fn();
const mockUseExperimentContainsTraces = jest.fn();
const mockUseGenAIOverviewIssueDetectionProgress = jest.fn();
const mockRegisterIssueDetectionJob = jest.fn();

jest.mock('../../../assistant', () => ({
  useAssistant: () => mockUseAssistant(),
}));

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableIssueDetection: () => mockShouldEnableIssueDetection(),
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<Record<string, unknown>>('@databricks/web-shared/genai-traces-table'),
  useSearchMlflowTraces: (...args: unknown[]) => mockUseSearchMlflowTraces(...args),
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

jest.mock('../experiment-page-tabs/SqlWarehouseContext', () => ({
  useSqlWarehouseContextSafe: () => null,
}));

jest.mock('./components/GenAIOverviewTraceActivityChart', () => ({
  GenAIOverviewTraceActivityChart: ({ onAskAssistant }: { onAskAssistant: (context: unknown) => void }) => (
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
  ),
}));

jest.mock('../../components/experiment-page/components/traces-v3/IssueDetectionModal', () => ({
  IssueDetectionModal: ({
    availableTraceIds,
    onJobStarted,
  }: {
    availableTraceIds: string[];
    onJobStarted: (job: {
      experimentId: string;
      jobId: string;
      runId: string;
      traceCount: number;
      submittedAtMs: number;
    }) => void;
  }) => (
    <div role="dialog" aria-label="Detect Issues">
      <span>{availableTraceIds.join(',')}</span>
      <button
        onClick={() =>
          onJobStarted({
            experimentId: '123',
            jobId: 'job-1',
            runId: 'run-1',
            traceCount: availableTraceIds.length,
            submittedAtMs: 1,
          })
        }
      >
        Start mocked detection
      </button>
    </div>
  ),
}));

describe('ExperimentGenAIJourneyOverviewPage', () => {
  const { history } = setupTestRouter();

  const renderPage = () =>
    renderWithDesignSystem(
      <TestRouter
        history={history}
        routes={[testRoute(<ExperimentGenAIJourneyOverviewPage />, RoutePaths.experimentPageTabJourneyOverview)]}
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
    mockUseSearchMlflowTraces.mockReturnValue({ data: [{ trace_id: 'trace-1' }, { trace_id: 'trace-2' }] });
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
      registerJob: mockRegisterIssueDetectionJob,
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

  test('prefills the Assistant when native issue detection is unavailable', async () => {
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'Detect Issues' }));

    expect(mockOpenAssistantPanel).toHaveBeenCalledTimes(1);
    expect(mockPrefillAssistantPrompt).toHaveBeenCalledWith(
      expect.stringContaining('Analyze these traces for systematic issues'),
    );
  });

  test('opens native issue detection with loaded traces and tracks the new job', async () => {
    mockShouldEnableIssueDetection.mockReturnValue(true);
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'Detect Issues' }));
    expect(await screen.findByRole('dialog', { name: 'Detect Issues' })).toHaveTextContent('trace-1,trace-2');
    await userEvent.click(screen.getByRole('button', { name: 'Start mocked detection' }));

    expect(mockRegisterIssueDetectionJob).toHaveBeenCalledWith({
      experimentId: '123',
      jobId: 'job-1',
      runId: 'run-1',
      traceCount: 2,
      submittedAtMs: 1,
    });
  });
});
