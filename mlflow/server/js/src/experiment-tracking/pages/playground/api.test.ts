import { jest, describe, it, expect, afterEach } from '@jest/globals';
import { PlaygroundApi } from './api';
import type { ChatCompletionRequest } from './types';

const mockRequest: ChatCompletionRequest = {
  model: 'my-endpoint',
  messages: [{ role: 'user', content: 'Hi' }],
};

const mockJsonResponse = (status: number, body: unknown) => {
  jest.spyOn(global, 'fetch').mockImplementation(() =>
    Promise.resolve(
      new Response(JSON.stringify(body), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  );
};

const callAndCatch = async (status: number) => {
  try {
    await PlaygroundApi.chatCompletion(mockRequest);
    throw new Error('expected chatCompletion to reject');
  } catch (e) {
    const err = e as { status?: number; message: string };
    expect(err.status).toBe(status);
    return err;
  }
};

describe('PlaygroundApi.chatCompletion', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('returns the parsed JSON body on success', async () => {
    const responseBody = {
      choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    };
    jest
      .spyOn(global, 'fetch')
      .mockImplementation(() => Promise.resolve(new Response(JSON.stringify(responseBody), { status: 200 })));
    await expect(PlaygroundApi.chatCompletion(mockRequest)).resolves.toEqual(responseBody);
  });

  it('extracts body.detail.message (MlflowException shape)', async () => {
    mockJsonResponse(400, { detail: { error_code: 'BAD_REQUEST', message: 'mlflow says no' } });
    const err = await callAndCatch(400);
    expect(err.message).toBe('mlflow says no');
  });

  it('extracts string body.detail (AIGatewayException shape)', async () => {
    mockJsonResponse(400, { detail: 'gemini said no' });
    const err = await callAndCatch(400);
    expect(err.message).toBe('gemini said no');
  });

  it('extracts top-level body.message', async () => {
    mockJsonResponse(500, { message: 'top-level message' });
    const err = await callAndCatch(500);
    expect(err.message).toBe('top-level message');
  });

  it('extracts body.error.message (passthrough provider shape)', async () => {
    mockJsonResponse(400, { error: { message: 'provider err' } });
    const err = await callAndCatch(400);
    expect(err.message).toBe('provider err');
  });

  it('falls back to the predefined default when no recognized field is present', async () => {
    mockJsonResponse(500, {});
    const err = await callAndCatch(500);
    expect(err.message).toEqual(expect.any(String));
    expect(err.message.length).toBeGreaterThan(0);
  });

  it('falls back when the response body is unparseable', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockImplementation(() => Promise.resolve(new Response('<html>not json</html>', { status: 500 })));
    const err = await callAndCatch(500);
    expect(err.message).toEqual(expect.any(String));
    expect(err.message.length).toBeGreaterThan(0);
  });

  it('ignores non-string detail values (e.g. detail: null)', async () => {
    mockJsonResponse(500, { detail: null });
    const err = await callAndCatch(500);
    expect(err.message).not.toBe('null');
  });
});
