import { MlflowClient } from '../../src/clients/client';
import { createAuthProvider } from '../../src/auth';
import { MlflowHttpError } from '../../src/clients/utils';

/**
 * Unit tests for {@link MlflowClient.exportOtlpSpans} used for OSS (non-Databricks)
 * tracking servers. These mock `global.fetch` so they exercise the request shaping
 * (endpoint, headers, body) and error mapping without needing a live tracking server.
 */
describe('MlflowClient.exportOtlpSpans', () => {
  const trackingUri = 'http://localhost:5000';
  let client: MlflowClient;
  let fetchMock: jest.Mock;
  const originalFetch = global.fetch;

  beforeEach(() => {
    const authProvider = createAuthProvider({ trackingUri });
    client = new MlflowClient({ trackingUri, authProvider });
    fetchMock = jest.fn().mockResolvedValue(new Response(null, { status: 200 }));
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('POSTs OTLP protobuf to the OSS endpoint with the protobuf content type and experiment header', async () => {
    const bytes = new Uint8Array([0x0a, 0x01, 0x02, 0x03]);

    await client.exportOtlpSpans('exp-123', bytes);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://localhost:5000/v1/traces');
    expect(init.method).toBe('POST');
    const headers = init.headers as Record<string, string>;
    expect(headers['Content-Type']).toBe('application/x-protobuf');
    expect(headers['x-mlflow-experiment-id']).toBe('exp-123');
    expect(init.body).toBe(bytes);
  });

  it('does not call fetch when there are no span bytes to send', async () => {
    await client.exportOtlpSpans('exp-123', new Uint8Array(0));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('surfaces a 501 as an MlflowHttpError so callers can fall back', async () => {
    fetchMock.mockResolvedValue(
      new Response('REST OTLP span logging is not supported', {
        status: 501,
        statusText: 'Not Implemented',
      }),
    );

    await expect(
      client.exportOtlpSpans('exp-123', new Uint8Array([0x0a, 0x01])),
    ).rejects.toBeInstanceOf(MlflowHttpError);
  });

  it('propagates the 501 status code on the thrown error', async () => {
    fetchMock.mockResolvedValue(
      new Response('nope', { status: 501, statusText: 'Not Implemented' }),
    );

    await expect(
      client.exportOtlpSpans('exp-123', new Uint8Array([0x0a, 0x01])),
    ).rejects.toMatchObject({
      status: 501,
    });
  });
});
