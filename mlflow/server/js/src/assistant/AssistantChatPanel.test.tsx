import { describe, test, expect, jest, beforeEach, beforeAll } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { AssistantChatPanel } from './AssistantChatPanel';
import { useLogTelemetryEvent } from '../telemetry/hooks/useLogTelemetryEvent';

jest.mock('../telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: jest.fn(() => jest.fn()),
}));

beforeAll(() => {
  // scrollIntoView is not available in JSDOM
  Element.prototype.scrollIntoView = jest.fn();
});

const mockSendMessage = jest.fn();
const mockCancelSession = jest.fn();
const mockClearPendingPrompt = jest.fn();
let mockSetupComplete = true;
let mockPendingPrompt: string | null = null;

jest.mock('./AssistantContext', () => ({
  useAssistant: () => ({
    isPanelOpen: true,
    sessionId: 'test-session',
    messages: [],
    isStreaming: false,
    error: null,
    currentStatus: null,
    activeTools: [],
    setupComplete: mockSetupComplete,
    isLoadingConfig: false,
    isLocalServer: true,
    pendingPrompt: mockPendingPrompt,
    openPanel: jest.fn(),
    closePanel: jest.fn(),
    sendMessage: mockSendMessage,
    prefillPrompt: jest.fn(),
    clearPendingPrompt: mockClearPendingPrompt,
    regenerateLastMessage: jest.fn(),
    reset: jest.fn(),
    cancelSession: mockCancelSession,
    refreshConfig: jest.fn(),
    completeSetup: jest.fn(),
  }),
}));

jest.mock('./AssistantPageContext', () => ({
  useAssistantPageContext: () => ({ experimentId: '123' }),
}));

jest.mock('../common/utils/RoutingUtils', () => ({
  useAssistantPrompts: () => ['Prompt 1', 'Prompt 2'],
}));

const renderChatPanel = () => {
  return renderWithIntl(
    <DesignSystemProvider>
      <AssistantChatPanel />
    </DesignSystemProvider>,
  );
};

describe('AssistantChatPanel', () => {
  let mockLogTelemetryEvent: jest.Mock;

  beforeEach(() => {
    mockSendMessage.mockClear();
    mockCancelSession.mockClear();
    mockClearPendingPrompt.mockClear();
    mockSetupComplete = true;
    mockPendingPrompt = null;
    mockLogTelemetryEvent = jest.fn();
    jest.mocked(useLogTelemetryEvent).mockReturnValue(mockLogTelemetryEvent);
  });

  test('when setup is NOT complete, the panel shows the "Get Started" setup prompt and no chat input', () => {
    mockSetupComplete = false;
    renderChatPanel();

    // The user is asked to set up the assistant ...
    expect(screen.getByRole('button', { name: 'Get Started' })).toBeInTheDocument();
    // ... and the chat input isn't mounted yet, so a queued prompt waits on the context.
    expect(screen.queryByPlaceholderText('Ask a question...')).not.toBeInTheDocument();
  });

  test('a queued pendingPrompt is dropped into the input once chat appears, then cleared', async () => {
    mockPendingPrompt = 'SEED';
    renderChatPanel();

    const textarea = await screen.findByDisplayValue('SEED');
    expect(textarea.tagName).toBe('TEXTAREA');
    expect(mockClearPendingPrompt).toHaveBeenCalledTimes(1);
  });

  // Not set up → complete setup WITHOUT closing → prompt prefilled.
  test('seed waits while setup is incomplete, then prefills the input after setup completes', async () => {
    mockSetupComplete = false;
    mockPendingPrompt = 'SEED';
    const { rerender } = renderChatPanel();

    // Setup prompt is shown; no input yet; the seed has NOT been consumed.
    expect(screen.getByRole('button', { name: 'Get Started' })).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Ask a question...')).not.toBeInTheDocument();
    expect(mockClearPendingPrompt).not.toHaveBeenCalled();

    // Setup completes (provider selected) — ChatPanelContent mounts and consumes the seed.
    mockSetupComplete = true;
    rerender(
      <DesignSystemProvider>
        <AssistantChatPanel />
      </DesignSystemProvider>,
    );

    const textarea = await screen.findByDisplayValue('SEED');
    expect(textarea.tagName).toBe('TEXTAREA');
    expect(mockClearPendingPrompt).toHaveBeenCalledTimes(1);
  });

  test('renders a textarea for chat input', () => {
    renderChatPanel();
    const textarea = screen.getByPlaceholderText('Ask a question...');
    expect(textarea.tagName).toBe('TEXTAREA');
  });

  test('Shift+Enter inserts a newline instead of sending', async () => {
    const user = userEvent.setup();
    renderChatPanel();
    const textarea = screen.getByPlaceholderText('Ask a question...');

    await user.click(textarea);
    await user.type(textarea, 'line one');
    await user.keyboard('{Shift>}{Enter}{/Shift}');
    await user.type(textarea, 'line two');

    expect(mockSendMessage).not.toHaveBeenCalled();
    expect(textarea).toHaveValue('line one\nline two');
  });

  test('Enter sends the message without Shift held', async () => {
    const user = userEvent.setup();
    renderChatPanel();
    const textarea = screen.getByPlaceholderText('Ask a question...');

    await user.click(textarea);
    await user.type(textarea, 'hello');
    await user.keyboard('{Enter}');

    expect(mockSendMessage).toHaveBeenCalledWith('hello');
  });

  test('Enter does not send when input is empty', async () => {
    const user = userEvent.setup();
    renderChatPanel();
    const textarea = screen.getByPlaceholderText('Ask a question...');

    await user.click(textarea);
    await user.keyboard('{Enter}');

    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  test('Enter logs a telemetry event when message is sent', async () => {
    const user = userEvent.setup();
    renderChatPanel();
    const textarea = screen.getByPlaceholderText('Ask a question...');

    await user.click(textarea);
    await user.type(textarea, 'hello');
    await user.keyboard('{Enter}');

    expect(mockLogTelemetryEvent).toHaveBeenCalledWith(
      expect.objectContaining({ componentId: 'mlflow.assistant.chat_panel.send' }),
    );
  });

  test('Enter does not log a telemetry event when input is empty', async () => {
    const user = userEvent.setup();
    renderChatPanel();
    const textarea = screen.getByPlaceholderText('Ask a question...');

    await user.click(textarea);
    await user.keyboard('{Enter}');

    expect(mockLogTelemetryEvent).not.toHaveBeenCalled();
  });
});
