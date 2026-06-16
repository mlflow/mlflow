import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { renderHook, act, cleanup } from '@testing-library/react';
import type { ReactNode } from 'react';

import { AssistantProvider, useAssistant } from './AssistantContext';
import * as AssistantService from './AssistantService';
import type { SendMessageStreamCallbacks } from './AssistantService';

jest.mock('./AssistantService', () => ({
  __esModule: true,
  sendMessageStream: jest.fn(),
  getConfig: jest.fn(),
  cancelSession: jest.fn(),
}));

const mockSendMessageStream = jest.mocked(AssistantService.sendMessageStream);
const mockGetConfig = jest.mocked(AssistantService.getConfig);

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
