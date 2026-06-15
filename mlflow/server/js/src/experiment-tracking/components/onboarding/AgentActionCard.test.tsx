import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

import { AgentActionCard } from './AgentActionCard';

const mockOpenPanel = jest.fn();
const mockSendMessage = jest.fn();
let mockSetupComplete = true;
let mockIsLocalServer = true;

jest.mock('../../../assistant', () => ({
  __esModule: true,
  useAssistant: () => ({
    openPanel: mockOpenPanel,
    sendMessage: mockSendMessage,
    setupComplete: mockSetupComplete,
    isPanelOpen: false,
    sessionId: null,
    messages: [],
    isStreaming: false,
    error: null,
    currentStatus: null,
    activeTools: [],
    isLoadingConfig: false,
    isLocalServer: mockIsLocalServer,
    closePanel: jest.fn(),
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
  mockSendMessage.mockClear();
  mockSetupComplete = true;
  mockIsLocalServer = true;
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

  it('hides the MLflow assistant tab when not on a local server', () => {
    mockIsLocalServer = false;
    renderCard();
    // Trigger (and its icon) gone...
    expect(screen.queryByRole('tab', { name: /MLflow assistant/ })).not.toBeInTheDocument();
    expect(screen.queryByTestId('assistant-sparkle-icon')).not.toBeInTheDocument();
    // ...and the Tab.Content too — the "Open assistant" button must not be in the DOM either.
    expect(screen.queryByText('Open assistant')).not.toBeInTheDocument();
  });

  it('hides the one-line setup tab by default and shows it when showAgentSetupTab is true', () => {
    const { rerender } = renderCard();
    expect(screen.queryByRole('tab', { name: /One-line setup/ })).not.toBeInTheDocument();

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
    expect(screen.getByRole('tab', { name: /One-line setup/ })).toBeInTheDocument();
  });

  it('clicking "Open assistant" with setup complete calls openPanel and sendMessage with the assistant prompt', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    await user.click(screen.getByRole('tab', { name: /MLflow assistant/ }));
    await user.click(screen.getByRole('button', { name: /Open assistant/ }));

    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).toHaveBeenCalledWith('ASSISTANT_PROMPT_BODY');
  });

  it('clicking "Open assistant" with setup NOT complete opens the panel but drops the prompt (known follow-up)', async () => {
    mockSetupComplete = false;
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    await user.click(screen.getByRole('tab', { name: /MLflow assistant/ }));
    await user.click(screen.getByRole('button', { name: /Open assistant/ }));

    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it('renders extraTabs after the built-in tabs and shows their content when selected', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard({
      extraTabs: [{ value: 'manual', label: 'Manual setup', content: <div>MANUAL_TAB_BODY</div> }],
    });

    const tabs = screen.getAllByRole('tab').map((t) => t.textContent?.trim() ?? '');
    expect(tabs).toEqual(['Copy for coding agent', 'MLflow assistant', 'Manual setup']);

    // Content is revealed only once the tab is selected.
    expect(screen.queryByText('MANUAL_TAB_BODY')).not.toBeInTheDocument();
    await user.click(screen.getByRole('tab', { name: /Manual setup/ }));
    expect(screen.getByText('MANUAL_TAB_BODY')).toBeInTheDocument();
  });
});
