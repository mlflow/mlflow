import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { resumeStream, sendMessageStream, processContentBlocks } from './AssistantService';
import type { ToolUseInfo, ToolResultInfo } from './types';

// jest.mock is hoisted above imports by babel-jest, so the mock still applies.
jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(() => Promise.resolve({})),
  getAjaxUrl: (url: string) => url,
  getDefaultHeaders: () => ({}),
}));

const mockedFetchAPI = jest.mocked(fetchAPI);

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static readonly CLOSED = 2;
  url: string;
  readyState = 0;
  listeners: Record<string, ((event: { data: string }) => void)[]> = {};

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, cb: (event: { data: string }) => void) {
    (this.listeners[type] ||= []).push(cb);
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }

  emit(type: string, data: unknown) {
    (this.listeners[type] || []).forEach((cb) => cb({ data: JSON.stringify(data) }));
  }
}

describe('AssistantService permissions', () => {
  beforeEach(() => {
    mockedFetchAPI.mockClear();
    FakeEventSource.instances = [];
    (global as any).EventSource = FakeEventSource;
  });

  test('resumeStream POSTs the decision then opens a fresh stream', async () => {
    const result = await resumeStream('sess-1', 'req-1', 'allow', {
      onMessage: jest.fn(),
      onError: jest.fn(),
      onDone: jest.fn(),
    });
    expect(mockedFetchAPI).toHaveBeenCalledWith('ajax-api/3.0/mlflow/assistant/sessions/sess-1/permission', {
      method: 'POST',
      body: JSON.stringify({ request_id: 'req-1', decision: 'allow' }),
    });
    expect(result.eventSource).toBeDefined();
    expect(FakeEventSource.instances).toHaveLength(1);
  });
});

describe('sendMessageStream permission_request event', () => {
  beforeEach(() => {
    FakeEventSource.instances = [];
    (global as any).EventSource = FakeEventSource;
    (global as any).fetch = jest.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve({ session_id: 'sess-1' }) }),
    );
  });

  test('a permission_request SSE event invokes onPermissionRequest and closes the stream', async () => {
    const onPermissionRequest = jest.fn();
    await sendMessageStream(
      { message: 'hi' },
      {
        onMessage: jest.fn(),
        onError: jest.fn(),
        onDone: jest.fn(),
        onPermissionRequest,
      },
    );

    const es = FakeEventSource.instances[0];
    expect(es).toBeDefined();
    es.emit('permission_request', { request_id: 'req-1', tool_name: 'Bash', tool_input: { command: 'ls' } });

    expect(onPermissionRequest).toHaveBeenCalledWith({
      sessionId: 'sess-1',
      requestId: 'req-1',
      toolName: 'Bash',
      toolInput: { command: 'ls' },
    });
    // The turn ends at the prompt: the stream is closed, the decision arrives via resumeStream.
    expect(es.readyState).toBe(FakeEventSource.CLOSED);
  });
});

describe('processContentBlocks', () => {
  test('emits text blocks via onMessage', () => {
    const onMessage = jest.fn();
    processContentBlocks([{ text: 'hello' }], onMessage);
    expect(onMessage).toHaveBeenCalledWith('hello');
  });

  test('emits tool_use blocks via onToolUse', () => {
    const onToolUse = jest.fn<(tools: ToolUseInfo[]) => void>();
    processContentBlocks(
      [{ id: 'tu1', name: 'Bash', input: { command: 'ls', description: 'list' } }],
      jest.fn(),
      onToolUse,
    );
    expect(onToolUse).toHaveBeenCalledWith([
      { id: 'tu1', name: 'Bash', description: 'list', input: { command: 'ls', description: 'list' } },
    ]);
  });

  test('emits a tool_result block via onToolResult with string content', () => {
    const onToolResult = jest.fn<(r: ToolResultInfo) => void>();
    processContentBlocks(
      [{ tool_use_id: 'tu1', content: 'done', is_error: false }],
      jest.fn(),
      jest.fn(),
      onToolResult,
    );
    expect(onToolResult).toHaveBeenCalledWith({ toolUseId: 'tu1', content: 'done', isError: false });
  });

  test('JSON-stringifies array tool_result content', () => {
    const onToolResult = jest.fn<(r: ToolResultInfo) => void>();
    processContentBlocks(
      [{ tool_use_id: 'tu1', content: [{ a: 1 }], is_error: true }],
      jest.fn(),
      jest.fn(),
      onToolResult,
    );
    expect(onToolResult).toHaveBeenCalledWith({
      toolUseId: 'tu1',
      content: JSON.stringify([{ a: 1 }], null, 2),
      isError: true,
    });
  });

  test('normalizes null tool_result content to an empty string', () => {
    const onToolResult = jest.fn<(r: ToolResultInfo) => void>();
    processContentBlocks([{ tool_use_id: 'tu1', content: null }], jest.fn(), jest.fn(), onToolResult);
    expect(onToolResult).toHaveBeenCalledWith({ toolUseId: 'tu1', content: '', isError: false });
  });

  test('routes a tool_result block to onToolResult, not onToolUse', () => {
    const onToolUse = jest.fn();
    const onToolResult = jest.fn();
    processContentBlocks(
      [{ tool_use_id: 'tu1', name: 'Bash', input: {}, content: 'x' }],
      jest.fn(),
      onToolUse,
      onToolResult,
    );
    expect(onToolResult).toHaveBeenCalledTimes(1);
    expect(onToolUse).not.toHaveBeenCalled();
  });
});
