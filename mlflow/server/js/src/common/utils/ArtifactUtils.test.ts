import { ErrorWrapper } from './ErrorWrapper';
import { afterEach, describe, expect, jest, test } from '@jest/globals';

import {
  getArtifactBlob,
  getArtifactChunkedText,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from './ArtifactUtils';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

describe('ArtifactUtils workspace-aware URLs', () => {
  afterEach(() => {
    setActiveWorkspace(null);
    jest.restoreAllMocks();
  });

  test('presigned artifact requests do not forward tracking-server headers', async () => {
    setActiveWorkspace('team-a');
    const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue(new Response('artifact'));

    await getArtifactBlob('https://storage.example/artifact', {
      headers: { 'x-storage-header': 'required' },
    });

    const request = fetchSpy.mock.calls[0][0] as Request;
    expect(request.headers.get('x-storage-header')).toBe('required');
    expect(request.headers.get('X-MLFLOW-WORKSPACE')).toBeNull();
  });

  test('artifact request options without headers retain tracking-server headers', async () => {
    setActiveWorkspace('team-a');
    const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue(new Response('artifact'));

    await getArtifactBlob('/get-artifact', { isBinary: true });

    const request = fetchSpy.mock.calls[0][0] as Request;
    expect(request.headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
  });

  test('getArtifactLocationUrl omits workspace segment and relies on headers', () => {
    setActiveWorkspace('team-a');
    const url = getArtifactLocationUrl('file.txt', 'run-123');

    expect(url).toContain('get-artifact');
    expect(url).not.toContain('workspaces');
    expect(url).toContain('path=file.txt');
    expect(url).toContain('run_uuid=run-123');
  });

  test('chunked text rejects browser transport errors', async () => {
    const error = new TypeError('Failed to fetch');
    jest.spyOn(global, 'fetch').mockRejectedValue(error);
    await expect(getArtifactChunkedText('https://storage.example/artifact', { headers: {} })).rejects.toBe(error);
  });

  test('chunked text rejects HTTP errors before reading the body', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(new Response('denied', { status: 403 }));
    await expect(getArtifactChunkedText('/get-artifact')).rejects.toMatchObject({ status: 403 });
  });

  test('chunked text decodes split UTF-8 characters and releases the reader', async () => {
    const read = jest
      .fn<() => Promise<ReadableStreamReadResult<Uint8Array>>>()
      .mockResolvedValueOnce({ value: new Uint8Array([0xe2, 0x82]), done: false })
      .mockResolvedValueOnce({ value: new Uint8Array([0xac]), done: false })
      .mockResolvedValueOnce({ value: undefined, done: true });
    const releaseLock = jest.fn();
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ ok: true, body: { getReader: () => ({ read, releaseLock }) } } as unknown as Response);
    await expect(getArtifactChunkedText('/get-artifact')).resolves.toBe('€');
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('chunked text propagates stream failures and releases the reader', async () => {
    const error = new TypeError('stream failed');
    const read = jest.fn<() => Promise<ReadableStreamReadResult<Uint8Array>>>().mockRejectedValue(error);
    const releaseLock = jest.fn();
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue({ ok: true, body: { getReader: () => ({ read, releaseLock }) } } as unknown as Response);
    await expect(getArtifactChunkedText('/get-artifact')).rejects.toBe(error);
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('getLoggedModelArtifactLocationUrl omits workspace segment and relies on headers', () => {
    setActiveWorkspace('team-b');
    const url = getLoggedModelArtifactLocationUrl('dir/file.txt', '42');

    expect(url).toContain('mlflow/logged-models/42/artifacts/files');
    expect(url).not.toContain('workspaces');
    expect(url).toContain('artifact_file_path=dir%2Ffile.txt');
  });
});

describe('getArtifactChunkedText', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  const streamedResponse = (chunks: string[]) => {
    const encoded = chunks.map((chunk) => new TextEncoder().encode(chunk));
    let index = 0;
    return {
      ok: true,
      body: {
        getReader: () => ({
          releaseLock: jest.fn(),
          cancel: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
          read: async () =>
            index < encoded.length ? { done: false, value: encoded[index++] } : { done: true, value: undefined },
        }),
      },
    } as unknown as Response;
  };

  test('joins the streamed chunks', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(streamedResponse(['hel', 'lo']));

    await expect(getArtifactChunkedText('artifact')).resolves.toBe('hello');
  });

  test('rejects when the request itself fails', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));

    await expect(getArtifactChunkedText('artifact')).rejects.toThrow('Failed to fetch');
  });

  test('rejects with the server error for a failed response', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      text: async () => 'missing',
    } as unknown as Response);

    const error = await getArtifactChunkedText('artifact').catch((e) => e);
    expect(error).toBeInstanceOf(ErrorWrapper);
    expect(error.status).toBe(404);
  });
});
