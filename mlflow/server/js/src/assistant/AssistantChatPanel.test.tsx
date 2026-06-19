import { describe, test, expect, jest, beforeEach, beforeAll } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { AssistantChatPanel, AssistantMessageBody, groupParts } from './AssistantChatPanel';
import type { AssistantPart, ChatMessage } from './types';
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

jest.mock('./AssistantContext', () => ({
  useAssistant: () => ({
    isPanelOpen: true,
    sessionId: 'test-session',
    messages: [],
    isStreaming: false,
    error: null,
    currentStatus: null,
    activeTools: [],
    setupComplete: true,
    isLoadingConfig: false,
    isLocalServer: true,
    tokenUsage: { promptTokens: 0, completionTokens: 0, totalTokens: 0, costUsd: null },
    openPanel: jest.fn(),
    closePanel: jest.fn(),
    sendMessage: mockSendMessage,
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

jest.mock('./AssistantComposerControls', () => ({
  AssistantComposerControls: () => null,
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
    mockLogTelemetryEvent = jest.fn();
    jest.mocked(useLogTelemetryEvent).mockReturnValue(mockLogTelemetryEvent);
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

const renderBody = (message: ChatMessage) =>
  renderWithIntl(
    <DesignSystemProvider>
      <AssistantMessageBody message={message} />
    </DesignSystemProvider>,
  );

const baseAssistantMessage = {
  id: 'm1',
  role: 'assistant' as const,
  content: '',
  timestamp: new Date(),
};

describe('groupParts', () => {
  test('coalesces adjacent tool calls while preserving interleaving order', () => {
    const parts: AssistantPart[] = [
      { type: 'text', text: 'a' },
      { type: 'toolCall', toolUseId: 't1', name: 'Bash' },
      { type: 'toolCall', toolUseId: 't2', name: 'Bash' },
      { type: 'text', text: 'b' },
      { type: 'toolCall', toolUseId: 't3', name: 'Read' },
    ];
    const groups = groupParts(parts);
    expect(groups).toEqual([
      { kind: 'text', text: 'a' },
      { kind: 'tools', calls: [parts[1], parts[2]] },
      { kind: 'text', text: 'b' },
      { kind: 'tools', calls: [parts[4]] },
    ]);
  });

  test('returns an empty array unchanged', () => {
    expect(groupParts([])).toEqual([]);
  });
});

describe('AssistantMessageBody', () => {
  test('renders text parts in order around a collapsed tool-call group', () => {
    renderBody({
      ...baseAssistantMessage,
      content: 'Found it.',
      parts: [
        { type: 'text', text: 'Let me check.' },
        {
          type: 'toolCall',
          toolUseId: 't1',
          name: 'trace_analyse',
          input: { trace_id: 'tr-1', jq_filter: '.data.spans' },
        },
        { type: 'text', text: 'Found it.' },
      ],
    });

    expect(screen.getByText('Let me check.')).toBeInTheDocument();
    expect(screen.getByText('Found it.')).toBeInTheDocument();
    // Collapsed: the header shows the count + tool name, but the inner card's input detail is hidden.
    expect(screen.getByText('1 tool call')).toBeInTheDocument();
    expect(screen.getByText('trace_analyse')).toBeInTheDocument();
    expect(screen.queryByText('tr-1 · .data.spans')).not.toBeInTheDocument();
  });

  test('expanding the group reveals the inner tool-call cards', async () => {
    const user = userEvent.setup();
    renderBody({
      ...baseAssistantMessage,
      parts: [{ type: 'toolCall', toolUseId: 't1', name: 'Bash', input: { command: 'mlflow traces search' } }],
    });

    // The command lives only in the inner card, hidden until the group is expanded.
    expect(screen.queryByText('mlflow traces search')).not.toBeInTheDocument();
    await user.click(screen.getByText('1 tool call'));
    expect(screen.getByText('mlflow traces search')).toBeInTheDocument();
  });

  test('falls back to content for legacy messages without parts', () => {
    renderBody({ ...baseAssistantMessage, content: 'legacy answer' });
    expect(screen.getByText('legacy answer')).toBeInTheDocument();
  });
});
