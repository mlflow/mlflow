import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { TextEncoder as NodeTextEncoder, TextDecoder as NodeTextDecoder } from 'util';

import { streamChatViaFetch } from './fetchStreamTransport';
import { type SendMessageStreamCallbacks } from './shared';

// jsdom doesn't provide TextEncoder/TextDecoder; the fetch reader (and these tests) need them.
if (typeof (global as any).TextDecoder === 'undefined') {
  (global as any).TextDecoder = NodeTextDecoder;
}
if (typeof (global as any).TextEncoder === 'undefined') {
  (global as any).TextEncoder = NodeTextEncoder;
}

const encoder = new NodeTextEncoder();
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

const makeCallbacks = (overrides: Partial<SendMessageStreamCallbacks> = {}): SendMessageStreamCallbacks => ({
  onMessage: jest.fn(),
  onError: jest.fn(),
  onDone: jest.fn(),
  onStatus: jest.fn(),
  onSessionId: jest.fn(),
  onToolUse: jest.fn(),
  onInterrupted: jest.fn(),
  onConversationHistory: jest.fn(),
  ...overrides,
});

// A reader that yields the given string chunks, then signals done.
const makeReader = (chunks: string[]) => {
  let i = 0;
  return {
    read: jest.fn(async () => {
      if (i < chunks.length) {
        const value = encoder.encode(chunks[i]);
        i += 1;
        return { value, done: false };
      }
      return { value: undefined, done: true };
    }),
  };
};

let mockFetch: jest.Mock<(...args: any[]) => Promise<any>>;

beforeEach(() => {
  mockFetch = jest.fn<(...args: any[]) => Promise<any>>();
  (global as any).fetch = mockFetch;
});

afterEach(() => {
  jest.restoreAllMocks();
  jest.clearAllMocks();
});

describe('streamChatViaFetch', () => {
  it('dispatches message and done callbacks, buffering a frame split across chunks', async () => {
    // The "Hello world" message frame is deliberately split across two reads so the
    // parser must hold the trailing partial frame until the next chunk completes it.
    const chunk1 = 'event: message\ndata: {"message":{"content":"Hello';
    const chunk2 = ' world"}}\n\nevent: done\ndata: {"session_id":"[BLOB]"}\n\n';
    mockFetch.mockResolvedValue({ ok: true, body: { getReader: () => makeReader([chunk1, chunk2]) } });

    const callbacks = makeCallbacks();
    await streamChatViaFetch({ message: 'hi' }, callbacks);
    await tick();

    expect(callbacks.onMessage).toHaveBeenCalledWith('Hello world');
    expect(callbacks.onConversationHistory).toHaveBeenCalledWith('[BLOB]');
    expect(callbacks.onDone).toHaveBeenCalledTimes(1);
    expect(callbacks.onError).not.toHaveBeenCalled();
  });

  it('dispatches content_delta and status stream events', async () => {
    const frames =
      'event: stream_event\ndata: {"event":{"type":"status","status":"Reading file"}}\n\n' +
      'event: stream_event\ndata: {"event":{"type":"content_delta","delta":{"text":"abc"}}}\n\n' +
      'event: done\ndata: {"session_id":"[]"}\n\n';
    mockFetch.mockResolvedValue({ ok: true, body: { getReader: () => makeReader([frames]) } });

    const callbacks = makeCallbacks();
    await streamChatViaFetch({ message: 'hi' }, callbacks);
    await tick();

    expect(callbacks.onStatus).toHaveBeenCalledWith('Reading file');
    expect(callbacks.onMessage).toHaveBeenCalledWith('abc');
    expect(callbacks.onDone).toHaveBeenCalledTimes(1);
  });

  it('sends the conversation_history in the POST body', async () => {
    mockFetch.mockResolvedValue({ ok: true, body: { getReader: () => makeReader(['event: done\ndata: {}\n\n']) } });

    await streamChatViaFetch({ message: 'hi', conversation_history: '[{"role":"system"}]' }, makeCallbacks());
    await tick();

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toMatchObject({
      message: 'hi',
      conversation_history: '[{"role":"system"}]',
    });
  });

  it('calls onError on a non-ok response', async () => {
    mockFetch.mockResolvedValue({ ok: false, body: null, text: async () => 'boom', statusText: 'Bad Request' });

    const callbacks = makeCallbacks();
    await streamChatViaFetch({ message: 'hi' }, callbacks);

    expect(callbacks.onError).toHaveBeenCalledWith(expect.stringContaining('boom'));
    expect(callbacks.onDone).not.toHaveBeenCalled();
  });

  it('does not surface an error when the stream is aborted', async () => {
    let rejectRead!: (reason: unknown) => void;
    const reader = {
      read: jest
        .fn<() => Promise<{ value: Uint8Array | undefined; done: boolean }>>()
        .mockResolvedValueOnce({
          value: encoder.encode('event: message\ndata: {"message":{"content":"hi"}}\n\n'),
          done: false,
        })
        .mockImplementationOnce(
          () =>
            new Promise((_resolve, reject) => {
              rejectRead = reject;
            }),
        ),
    };
    mockFetch.mockResolvedValue({ ok: true, body: { getReader: () => reader } });

    const callbacks = makeCallbacks();
    const result = await streamChatViaFetch({ message: 'hi' }, callbacks);
    await tick();

    result.cancel(); // aborts the controller → signal.aborted = true
    rejectRead(new DOMException('aborted', 'AbortError'));
    await tick();

    expect(callbacks.onMessage).toHaveBeenCalledWith('hi');
    expect(callbacks.onError).not.toHaveBeenCalled();
  });

  it('treats a done after a permission_request as a pause: captures history without finalizing', async () => {
    // The provider emits permission_request then a done carrying the paused history (with the
    // unresolved tool_call). The turn must NOT finalize — the prompt stays up for an allow/deny.
    const frames =
      'event: permission_request\ndata: {"request_id":"req-1","tool_name":"Bash","tool_input":{"command":"ls"}}\n\n' +
      'event: done\ndata: {"session_id":"[PAUSED]"}\n\n';
    mockFetch.mockResolvedValue({ ok: true, body: { getReader: () => makeReader([frames]) } });

    const onPermissionRequest = jest.fn();
    const callbacks = makeCallbacks({ onPermissionRequest });
    await streamChatViaFetch({ message: 'run tool' }, callbacks);
    await tick();

    expect(onPermissionRequest).toHaveBeenCalledWith({
      requestId: 'req-1',
      toolName: 'Bash',
      toolInput: { command: 'ls' },
    });
    // The paused history blob is captured for the resume...
    expect(callbacks.onConversationHistory).toHaveBeenCalledWith('[PAUSED]');
    // ...but the turn is NOT finalized (no onDone), so the prompt stays and isStreaming holds.
    expect(callbacks.onDone).not.toHaveBeenCalled();
    // The pause is still terminal for the read loop, so no "ended unexpectedly" error.
    expect(callbacks.onError).not.toHaveBeenCalled();
  });

  it('sends tool_decisions in the POST body when resuming a paused turn', async () => {
    mockFetch.mockResolvedValue({ ok: true, body: { getReader: () => makeReader(['event: done\ndata: {}\n\n']) } });

    await streamChatViaFetch(
      { message: '', conversation_history: '[]', tool_decisions: { 'call-1': 'allow' } },
      makeCallbacks(),
    );
    await tick();

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toMatchObject({ tool_decisions: { 'call-1': 'allow' } });
  });
});
