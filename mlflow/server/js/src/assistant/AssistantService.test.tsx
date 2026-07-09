import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { resumeStream, sendMessageStream } from './AssistantService';

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
