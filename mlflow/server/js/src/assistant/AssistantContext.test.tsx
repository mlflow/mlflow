import { describe, it, test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { renderHook, act, cleanup, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

import { AssistantProvider, useAssistant, reviveMessages, trimForStorage, CHAT_STORAGE_KEY } from './AssistantContext';
import * as AssistantService from './AssistantService';
import type { SendMessageStreamCallbacks } from './AssistantService';
import { GatewayApi } from '../gateway/api';
import type { AssistantConfig, ChatMessage, ProviderConfig } from './types';

const makeMessage = (overrides: Partial<ChatMessage> = {}): ChatMessage => ({
  id: 'msg-1',
  role: 'user',
  content: 'hello',
  timestamp: new Date('2026-01-01T00:00:00.000Z'),
  ...overrides,
});

jest.mock('./AssistantService', () => ({
  __esModule: true,
  sendMessageStream: jest.fn(),
  getConfig: jest.fn(),
  cancelSession: jest.fn(),
}));

jest.mock('../gateway/api', () => ({
  GatewayApi: { listEndpoints: jest.fn() },
}));

jest.mock('./AssistantPageContext', () => ({
  useAssistantPageContextActions: () => ({ getContext: () => ({}) }),
}));

const mockSendMessageStream = jest.mocked(AssistantService.sendMessageStream);
const mockGetConfig = jest.mocked(AssistantService.getConfig);
const mockListEndpoints = jest.mocked(GatewayApi.listEndpoints);

// A fake EventSource — the real one is created inside sendMessageStream, which we mock,
// so the context only ever calls .close() on what we hand back here.
let fakeEventSource: { close: jest.Mock };
// Capture the callbacks the context passes in so a test can simulate the backend streaming.
let capturedCallbacks: SendMessageStreamCallbacks | undefined;

const wrapper = ({ children }: { children: ReactNode }) => <AssistantProvider>{children}</AssistantProvider>;

// Render and flush the mount-time refreshConfig() promise so its state update lands inside act().
const renderAssistant = async () => {
  const utils = renderHook(() => useAssistant(), { wrapper });
  await act(async () => {});
  return utils;
};

beforeEach(() => {
  localStorage.clear();
  fakeEventSource = { close: jest.fn() };
  capturedCallbacks = undefined;
  mockGetConfig.mockResolvedValue({ providers: {}, projects: {} });
  mockSendMessageStream.mockImplementation(async (_req, callbacks) => {
    capturedCallbacks = callbacks;
    return { eventSource: fakeEventSource as unknown as EventSource };
  });
  // Control rAF so a scheduled flush stays pending until we assert on it.
  jest.spyOn(window, 'requestAnimationFrame').mockReturnValue(777 as unknown as number);
  jest.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  jest.restoreAllMocks();
  jest.clearAllMocks();
});

describe('AssistantContext — reset() tears down the active stream', () => {
  it('closes the active EventSource when reset() is called mid-stream', async () => {
    const { result } = await renderAssistant();

    // Start a stream via the public sendMessage path (no session yet → startChat).
    await act(async () => {
      result.current.sendMessage('hello');
    });
    expect(fakeEventSource.close).not.toHaveBeenCalled();

    act(() => {
      result.current.reset();
    });

    expect(fakeEventSource.close).toHaveBeenCalledTimes(1);
  });

  it('cancels a pending animation frame when reset() is called', async () => {
    const { result } = await renderAssistant();

    await act(async () => {
      result.current.sendMessage('hello');
    });

    // Simulate a token arriving — this schedules a rAF flush (id 777).
    act(() => {
      capturedCallbacks?.onMessage('partial token');
    });
    expect(window.requestAnimationFrame).toHaveBeenCalled();

    act(() => {
      result.current.reset();
    });

    expect(window.cancelAnimationFrame).toHaveBeenCalledWith(777);
  });

  it('clears the session so the next turn starts fresh', async () => {
    const { result } = await renderAssistant();

    await act(async () => {
      result.current.sendMessage('hello');
    });
    act(() => {
      capturedCallbacks?.onSessionId?.('session-OLD');
    });
    expect(result.current.sessionId).toBe('session-OLD');

    act(() => {
      result.current.reset();
    });

    expect(result.current.sessionId).toBeNull();
    expect(result.current.messages).toHaveLength(0);
  });
});

describe('AssistantContext — reset() during the in-flight send window', () => {
  // Drive sendMessageStream with a deferred promise so reset() can run while the POST is still
  // pending — the exact window where the captured token is invalidated before the stream attaches.
  const deferSend = () => {
    let resolveSend!: (result: { eventSource: EventSource | null }) => void;
    mockSendMessageStream.mockImplementation((_req, callbacks) => {
      capturedCallbacks = callbacks;
      return new Promise((resolve) => {
        resolveSend = resolve;
      });
    });
    return { resolve: () => resolveSend({ eventSource: fakeEventSource as unknown as EventSource }) };
  };

  it('ignores a stale onSessionId fired after reset() (no session revival)', async () => {
    const { result } = await renderAssistant();
    deferSend();

    // Start the send but leave the POST pending (do not await it).
    act(() => {
      result.current.sendMessage('hello');
    });

    act(() => {
      result.current.reset();
    });

    // The backend reply lands after reset — the guarded callback must no-op.
    act(() => {
      capturedCallbacks?.onSessionId?.('session-OLD');
    });

    expect(result.current.sessionId).toBeNull();
  });

  it('closes the late-resolved EventSource instead of storing it', async () => {
    const { result } = await renderAssistant();
    const send = deferSend();

    act(() => {
      result.current.sendMessage('hello');
    });
    act(() => {
      result.current.reset();
    });

    // POST resolves after reset: the post-await guard closes the orphaned stream.
    await act(async () => {
      send.resolve();
    });
    expect(fakeEventSource.close).toHaveBeenCalledTimes(1);

    // A second reset() must not close it again — proving it was never stored in eventSourceRef.
    act(() => {
      result.current.reset();
    });
    expect(fakeEventSource.close).toHaveBeenCalledTimes(1);
  });

  it('closes a regenerate stream orphaned by reset() during its in-flight window', async () => {
    const { result } = await renderAssistant();

    // Seed a completed turn so regenerateLastMessage has a user message to replay.
    const firstSend = deferSend();
    act(() => {
      result.current.sendMessage('hello');
    });
    await act(async () => {
      firstSend.resolve();
    });
    act(() => {
      capturedCallbacks?.onSessionId?.('session-1');
      capturedCallbacks?.onDone();
    });
    expect(result.current.isStreaming).toBe(false);
    fakeEventSource.close.mockClear();

    // Regenerate, but leave its POST pending, then reset before it attaches.
    const regen = deferSend();
    act(() => {
      result.current.regenerateLastMessage();
    });
    act(() => {
      result.current.reset();
    });

    await act(async () => {
      regen.resolve();
    });

    // The orphaned regenerate stream is closed by the guard, and the session stays cleared.
    expect(fakeEventSource.close).toHaveBeenCalledTimes(1);
    expect(result.current.sessionId).toBeNull();
  });
});

describe('AssistantContext — pendingPrompt seed', () => {
  it('prefillPrompt sets pendingPrompt and clearPendingPrompt nulls it', async () => {
    const { result } = await renderAssistant();
    expect(result.current.pendingPrompt).toBeNull();

    act(() => {
      result.current.prefillPrompt('SEED');
    });
    expect(result.current.pendingPrompt).toBe('SEED');

    act(() => {
      result.current.clearPendingPrompt();
    });
    expect(result.current.pendingPrompt).toBeNull();
  });

  it('closePanel clears a queued pendingPrompt (abandon ⇒ no stale inject later)', async () => {
    const { result } = await renderAssistant();

    act(() => {
      result.current.prefillPrompt('SEED');
    });
    expect(result.current.pendingPrompt).toBe('SEED');

    act(() => {
      result.current.closePanel();
    });
    expect(result.current.pendingPrompt).toBeNull();
  });

  // completing setup must NOT drop a queued prompt, so it can be
  // consumed once the chat input appears post-setup.
  it('keeps pendingPrompt across completeSetup() (seed survives the setup wizard)', async () => {
    const { result } = await renderAssistant();

    act(() => {
      result.current.prefillPrompt('SEED');
    });
    expect(result.current.pendingPrompt).toBe('SEED');

    // completeSetup() re-fetches config; mirror a finished wizard where a provider is selected
    // so setupComplete stays true after the refresh lands.
    mockGetConfig.mockResolvedValue({
      providers: { anthropic: { model: 'm', selected: true, permissions: {} } },
      projects: {},
    } as unknown as Awaited<ReturnType<typeof AssistantService.getConfig>>);

    await act(async () => {
      result.current.completeSetup();
    });

    expect(result.current.setupComplete).toBe(true);
    expect(result.current.pendingPrompt).toBe('SEED');
  });
});

const providerConfig = (overrides: Partial<ProviderConfig>): ProviderConfig => ({
  model: 'default',
  selected: false,
  permissions: { allow_edit_files: true, allow_read_docs: true, full_access: false },
  ...overrides,
});

const config = (providers: AssistantConfig['providers']): AssistantConfig => ({
  providers,
  projects: {},
});

describe('AssistantProvider setup completeness', () => {
  const renderAndWaitForConfig = async () => {
    const { result } = renderHook(() => useAssistant(), { wrapper: AssistantProvider });
    await waitFor(() => expect(result.current.isLoadingConfig).toBe(false));
    return result;
  };

  beforeEach(() => {
    mockGetConfig.mockReset();
    mockListEndpoints.mockReset();
  });

  test('gateway selected but no endpoints exist => setup incomplete', async () => {
    mockGetConfig.mockResolvedValue(config({ mlflow_gateway: providerConfig({ model: 'assistant', selected: true }) }));
    mockListEndpoints.mockResolvedValue({ endpoints: [] });

    const result = await renderAndWaitForConfig();

    expect(result.current.setupComplete).toBe(false);
  });

  test('gateway selected but configured endpoint is missing from the list => setup incomplete', async () => {
    mockGetConfig.mockResolvedValue(config({ mlflow_gateway: providerConfig({ model: 'assistant', selected: true }) }));
    mockListEndpoints.mockResolvedValue({ endpoints: [{ name: 'some-other-endpoint' }] as any });

    const result = await renderAndWaitForConfig();

    expect(result.current.setupComplete).toBe(false);
  });

  test('gateway selected and configured endpoint exists => setup complete', async () => {
    mockGetConfig.mockResolvedValue(config({ mlflow_gateway: providerConfig({ model: 'assistant', selected: true }) }));
    mockListEndpoints.mockResolvedValue({ endpoints: [{ name: 'assistant' }] as any });

    const result = await renderAndWaitForConfig();

    expect(result.current.setupComplete).toBe(true);
    expect(mockListEndpoints).toHaveBeenCalled();
  });

  test('non-gateway provider selected => setup complete without querying gateway endpoints', async () => {
    mockGetConfig.mockResolvedValue(config({ claude_code: providerConfig({ model: 'default', selected: true }) }));

    const result = await renderAndWaitForConfig();

    expect(result.current.setupComplete).toBe(true);
    expect(mockListEndpoints).not.toHaveBeenCalled();
  });
});

describe('AssistantContext — a new message supersedes a pending permission prompt', () => {
  // The pause path surfaces the request and closes the stream WITHOUT a done event,
  // so the Allow/Deny prompt is left showing.
  const pausePrompt = () => {
    capturedCallbacks?.onPermissionRequest?.({
      sessionId: 'session-1',
      requestId: 'req-1',
      toolName: 'bash',
      toolInput: { command: 'ls' },
    });
  };

  it('clears pendingPermission on the cold-start path (startChat, no session yet)', async () => {
    const { result } = await renderAssistant();

    await act(async () => {
      result.current.sendMessage('run the tool');
    });
    act(pausePrompt);
    expect(result.current.pendingPermission).not.toBeNull();

    // No session was established, so this send falls through to startChat.
    await act(async () => {
      result.current.sendMessage('never mind, what is 2+2');
    });

    expect(result.current.pendingPermission).toBeNull();
  });

  it('clears pendingPermission on the established-session path (handleSendMessage)', async () => {
    const { result } = await renderAssistant();

    await act(async () => {
      result.current.sendMessage('run the tool');
    });

    // The first turn returns a session id, so subsequent sends route through
    // handleSendMessage's own branch rather than startChat. Then the turn pauses.
    act(() => {
      capturedCallbacks?.onSessionId?.('session-1');
      pausePrompt();
    });
    expect(result.current.sessionId).toBe('session-1');
    expect(result.current.pendingPermission).not.toBeNull();

    // This send exercises handleSendMessage (sessionId is set); the stale prompt must clear.
    await act(async () => {
      result.current.sendMessage('never mind, what is 2+2');
    });

    expect(result.current.pendingPermission).toBeNull();
  });
});

describe('reviveMessages', () => {
  it('restores a JSON-stringified timestamp back to a Date', () => {
    const serialized = JSON.parse(JSON.stringify([makeMessage()])) as ChatMessage[];
    expect(typeof serialized[0].timestamp).toBe('string');

    const revived = reviveMessages(serialized);

    expect(revived[0].timestamp).toBeInstanceOf(Date);
    expect(revived[0].timestamp.toISOString()).toBe('2026-01-01T00:00:00.000Z');
  });

  it('preserves all other message fields', () => {
    const revived = reviveMessages([
      makeMessage({ id: 'msg-9', role: 'assistant', content: 'hi', isInterrupted: true }),
    ]);
    expect(revived[0]).toMatchObject({ id: 'msg-9', role: 'assistant', content: 'hi', isInterrupted: true });
  });

  it('returns an empty array unchanged', () => {
    expect(reviveMessages([])).toEqual([]);
  });
});

describe('trimForStorage', () => {
  it('returns the transcript unchanged when under the byte budget', () => {
    const messages = [makeMessage({ id: 'a' }), makeMessage({ id: 'b' })];
    expect(trimForStorage(messages, 1_000_000)).toEqual(messages);
  });

  it('drops the oldest messages until under the byte budget', () => {
    const messages = [
      makeMessage({ id: 'a', content: 'x'.repeat(200) }),
      makeMessage({ id: 'b', content: 'x'.repeat(200) }),
      makeMessage({ id: 'c', content: 'x'.repeat(200) }),
    ];

    const trimmed = trimForStorage(messages, 500);

    expect(trimmed.length).toBeLessThan(messages.length);
    // The newest message is always kept; the oldest is dropped first.
    expect(trimmed[trimmed.length - 1].id).toBe('c');
    expect(trimmed.map((m) => m.id)).not.toContain('a');
  });

  it('never drops the last remaining message even if it exceeds the budget', () => {
    const messages = [makeMessage({ id: 'only', content: 'x'.repeat(1000) })];
    expect(trimForStorage(messages, 10)).toEqual(messages);
  });
});

describe('AssistantContext — localStorage chat persistence', () => {
  it('restores messages from localStorage on mount', async () => {
    localStorage.setItem(
      CHAT_STORAGE_KEY,
      JSON.stringify({ messages: [makeMessage({ id: 'restored', content: 'from storage' })] }),
    );

    const { result } = await renderAssistant();

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0]).toMatchObject({ id: 'restored', content: 'from storage' });
    // timestamp must be revived to a Date, not left as a string.
    expect(result.current.messages[0].timestamp).toBeInstanceOf(Date);
  });

  it('persists a sent message to localStorage once streaming settles', async () => {
    const { result } = await renderAssistant();

    await act(async () => {
      result.current.sendMessage('persist me');
    });
    // Finish the stream so the persistence effect runs on the settled state.
    await act(async () => {
      capturedCallbacks?.onDone();
    });

    const stored = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{}');
    expect(stored.messages.some((m: ChatMessage) => m.content === 'persist me')).toBe(true);
  });

  it('clears persisted messages when reset() is called', async () => {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ messages: [makeMessage({ id: 'restored' })] }));

    const { result } = await renderAssistant();
    expect(result.current.messages).toHaveLength(1);

    act(() => {
      result.current.reset();
    });

    expect(result.current.messages).toHaveLength(0);
    const stored = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{}');
    expect(stored.messages).toEqual([]);
  });

  it('persists the interrupted turn when a stream is cancelled mid-stream', async () => {
    // Capture the scheduled rAF flush so a streamed token actually lands in the
    // assistant message before we cancel (the mount mock no-ops rAF otherwise).
    let rafFlush: FrameRequestCallback | undefined;
    jest.mocked(window.requestAnimationFrame).mockImplementation((cb: FrameRequestCallback) => {
      rafFlush = cb;
      return 777 as unknown as number;
    });

    // handleCancelSession fires the backend cancel API and .catch()es it; give it a resolved promise.
    jest.mocked(AssistantService.cancelSession).mockResolvedValue({ message: 'cancelled' });

    const { result } = await renderAssistant();

    await act(async () => {
      result.current.sendMessage('cancel me');
    });
    // cancelSession guards on a known sessionId, so the backend must report one first.
    act(() => {
      capturedCallbacks?.onSessionId?.('session-cancel');
    });
    // Deliver a partial token and flush it so the assistant message has real content.
    act(() => {
      capturedCallbacks?.onMessage('partial answer');
      rafFlush?.(0);
    });

    act(() => {
      result.current.cancelSession();
    });

    const stored = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{}');
    expect(stored.messages.some((m: ChatMessage) => m.content === 'cancel me')).toBe(true);
    const interrupted = stored.messages.find((m: ChatMessage) => m.role === 'assistant');
    expect(interrupted).toMatchObject({ isInterrupted: true, content: 'partial answer' });
  });

  it('does not write the in-flight turn to localStorage while streaming', async () => {
    const { result } = await renderAssistant();

    await act(async () => {
      result.current.sendMessage('still streaming');
    });
    // Establish a session and stream a token, but never settle the turn.
    act(() => {
      capturedCallbacks?.onSessionId?.('session-inflight');
      capturedCallbacks?.onMessage('partial answer');
    });

    // The mount effect may have written an empty transcript; the in-flight user
    // message must not be persisted while isStreaming is still true.
    const stored = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) ?? '{"messages":[]}');
    expect(stored.messages.some((m: ChatMessage) => m.content === 'still streaming')).toBe(false);
    expect(result.current.isStreaming).toBe(true);
  });
});
