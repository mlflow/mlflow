import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

import { EvalRunsEmptyStateCard } from './EvalRunsEmptyStateCard';

const mockOpenPanel = jest.fn();
const mockQueueMessage = jest.fn();

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  __esModule: true,
  useAssistant: () => ({
    openPanel: mockOpenPanel,
    queueMessage: mockQueueMessage,
    isPanelOpen: false,
    sessionId: null,
    messages: [],
    isStreaming: false,
    error: null,
    currentStatus: null,
    activeTools: [],
    setupComplete: false,
    isLoadingConfig: false,
    isLocalServer: true,
    closePanel: jest.fn(),
    sendMessage: jest.fn(),
    regenerateLastMessage: jest.fn(),
    reset: jest.fn(),
    cancelSession: jest.fn(),
    refreshConfig: jest.fn(),
    completeSetup: jest.fn(),
  }),
  AssistantSparkleIcon: ({ iconSize }: { iconSize?: number }) => (
    <span data-testid="assistant-sparkle-icon" data-icon-size={iconSize} />
  ),
}));

// RunEvaluationButton renders a primary "Evaluate traces" button. We stub it so we don't
// have to set up the trace-search + scorer-fetch hooks just to test the empty-state shell.
jest.mock('./RunEvaluationButton', () => ({
  __esModule: true,
  RunEvaluationButton: ({ label, type }: { label?: React.ReactNode; type?: string }) => (
    <button data-testid="run-evaluation-button" data-button-type={type}>
      {label}
    </button>
  ),
}));

const renderCard = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <EvalRunsEmptyStateCard experimentId="42" />
    </DesignSystemProvider>,
  );

beforeEach(() => {
  mockOpenPanel.mockClear();
  mockQueueMessage.mockClear();
});

describe('EvalRunsEmptyStateCard', () => {
  it('renders the primary "Evaluate traces" CTA at the top', () => {
    renderCard();
    const btn = screen.getByTestId('run-evaluation-button');
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveAttribute('data-button-type', 'primary');
    expect(btn).toHaveTextContent('Evaluate traces');
  });

  it('renders the three tabs in order: Your coding agent → Python → MLflow assistant', () => {
    renderCard();
    const tabs = screen.getAllByRole('tab').map((t) => t.textContent?.trim() ?? '');
    expect(tabs).toEqual(['Your coding agent', 'Python', 'MLflow assistant']);
  });

  it('renders the sparkle icon on the MLflow assistant tab', () => {
    renderCard();
    expect(screen.getByTestId('assistant-sparkle-icon')).toBeInTheDocument();
  });

  it('clicking "Open assistant" calls openPanel and queueMessage with the eval assistant prompt', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    await user.click(screen.getByRole('tab', { name: /MLflow assistant/ }));
    await user.click(screen.getByRole('button', { name: /Open assistant/ }));

    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
    expect(mockQueueMessage).toHaveBeenCalledTimes(1);
    expect(mockQueueMessage.mock.calls[0][0]).toContain('experiment ID: 42');
  });
});
