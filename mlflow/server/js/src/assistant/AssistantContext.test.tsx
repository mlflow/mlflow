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
