import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

import { AgentActionCard } from './AgentActionCard';

const mockOpenPanel = jest.fn();
const mockQueueMessage = jest.fn();

jest.mock('../../../assistant', () => ({
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

const renderCard = (props: Partial<React.ComponentProps<typeof AgentActionCard>> = {}) =>
  renderWithIntl(
    <DesignSystemProvider>
      <AgentActionCard
        title="Get help with X"
        codingAgentPrompt="CODING_AGENT_PROMPT_BODY"
        assistantPrompt="ASSISTANT_PROMPT_BODY"
        componentId="test.agent-card"
        {...props}
      />
    </DesignSystemProvider>,
  );

beforeEach(() => {
  mockOpenPanel.mockClear();
  mockQueueMessage.mockClear();
});

describe('AgentActionCard', () => {
  it('renders the card title', () => {
    renderCard();
    expect(screen.getByText('Get help with X')).toBeInTheDocument();
  });

  it('renders the MLflow assistant tab with a sparkle icon', () => {
    renderCard();
    const assistantTab = screen.getByRole('tab', { name: /MLflow assistant/ });
    expect(assistantTab).toBeInTheDocument();
    // Sparkle should be inside the assistant trigger (only one in the card now).
    expect(screen.getByTestId('assistant-sparkle-icon')).toBeInTheDocument();
  });

  it('hides the CLI tab by default and shows it when showAgentSetupTab is true', () => {
    const { rerender } = renderCard();
    expect(screen.queryByRole('tab', { name: 'CLI' })).not.toBeInTheDocument();

    rerender(
      <DesignSystemProvider>
        <AgentActionCard
          title="t"
          codingAgentPrompt="x"
          assistantPrompt="y"
          componentId="test.agent-card"
          showAgentSetupTab
        />
      </DesignSystemProvider>,
    );
    expect(screen.getByRole('tab', { name: 'CLI' })).toBeInTheDocument();
  });

  it('clicking "Open assistant" calls openPanel and queueMessage with the assistant prompt', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    // Switch to the assistant tab so the button is in the DOM.
    await user.click(screen.getByRole('tab', { name: /MLflow assistant/ }));
    await user.click(screen.getByRole('button', { name: /Open assistant/ }));

    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
    expect(mockQueueMessage).toHaveBeenCalledTimes(1);
    expect(mockQueueMessage).toHaveBeenCalledWith('ASSISTANT_PROMPT_BODY');
  });
});
