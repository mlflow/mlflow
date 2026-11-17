import { ErrorWrapper } from './ErrorWrapper';
import { afterEach, describe, expect, jest, test } from '@jest/globals';

import {
  getArtifactBlob,
  getArtifactChunkedText,
  getArtifactContent,
  getArtifactBytesContent,
  getArtifactDownloadBlob,
  MAX_ARTIFACT_PREVIEW_BYTES,
  ArtifactPreviewTooLargeError,
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
    const { fetchSpy } = mockArtifactStream([new Uint8Array([1])]);

    await getArtifactBlob('https://storage.example/artifact', {
      headers: { 'x-storage-header': 'required' },
    });

    const request = fetchSpy.mock.calls[0][0] as Request;
    expect(request.headers.get('x-storage-header')).toBe('required');
    expect(request.headers.get('X-MLFLOW-WORKSPACE')).toBeNull();
  });

  test('artifact request options without headers retain tracking-server headers', async () => {
    setActiveWorkspace('team-a');
    const { fetchSpy } = mockArtifactStream([new Uint8Array([1])]);

    await getArtifactBlob('/get-artifact', {});

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
    const cancel = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      body: { getReader: () => ({ read, releaseLock, cancel }) },
    } as unknown as Response);
    await expect(getArtifactChunkedText('/get-artifact')).resolves.toBe('€');
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('chunked text propagates stream failures and releases the reader', async () => {
    const error = new TypeError('stream failed');
    const read = jest.fn<() => Promise<ReadableStreamReadResult<Uint8Array>>>().mockRejectedValue(error);
    const releaseLock = jest.fn();
    const cancel = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      body: { getReader: () => ({ read, releaseLock, cancel }) },
    } as unknown as Response);
    await expect(getArtifactChunkedText('/get-artifact')).rejects.toBe(error);
    expect(cancel).toHaveBeenCalledTimes(1);
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

const mockArtifactStream = (chunks: Uint8Array[], headers: Record<string, string> = {}) => {
  let position = 0;
  const read = jest.fn<() => Promise<ReadableStreamReadResult<Uint8Array>>>().mockImplementation(async () => {
    if (position === chunks.length) {
      return { done: true, value: undefined };
    }
    return { done: false, value: chunks[position++] };
  });
  const cancel = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
  const releaseLock = jest.fn();
  const response = {
    ok: true,
    headers: new Headers(headers),
    body: { getReader: () => ({ read, cancel, releaseLock }) },
  } as unknown as Response;
  const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue(response);
  return { read, cancel, releaseLock, fetchSpy };
};

describe('bounded artifact previews', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  test.each([
    ['UTF-8', [0x41, 0xe2, 0x82, 0xac], 'text/plain', 'A€'],
    ['UTF-8 BOM', [0xef, 0xbb, 0xbf, 0x41, 0xe2, 0x82, 0xac], 'text/plain', 'A€'],
    ['UTF-16LE BOM', [0xff, 0xfe, 0x41, 0x00, 0xac, 0x20, 0x3d, 0xd8, 0x00, 0xde], 'text/plain', 'A€😀'],
    ['UTF-16BE BOM', [0xfe, 0xff, 0x00, 0x41, 0x20, 0xac, 0xd8, 0x3d, 0xde, 0x00], 'text/plain', 'A€😀'],
    ['BOM overriding charset', [0xff, 0xfe, 0x41, 0x00], 'text/plain;charset=utf-8', 'A'],
    ['empty text', [], 'text/plain', ''],
  ] as const)('preserves %s decoding across split stream chunks', async (_encoding, values, contentType, text) => {
    const bytes = new Uint8Array(values);
    const { fetchSpy, releaseLock } = mockArtifactStream([bytes.slice(0, 1), bytes.slice(1, 3), bytes.slice(3)], {
      'Content-Type': contentType,
    });

    await expect(
      getArtifactContent('https://storage.example/artifact', { headers: { 'x-storage': 'required' } }),
    ).resolves.toBe(text);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect((fetchSpy.mock.calls[0][0] as Request).headers.get('x-storage')).toBe('required');
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test.each([
    ['text', getArtifactContent],
    ['bytes', getArtifactBytesContent],
    ['blob', getArtifactBlob],
  ])('%s cancels a preview after 50 MiB without reading the remaining response', async (_, readArtifact) => {
    const { read, cancel, releaseLock } = mockArtifactStream([
      new Uint8Array([0xc2, 0xa2]),
      new Uint8Array(MAX_ARTIFACT_PREVIEW_BYTES - 1),
      new Uint8Array([1]),
    ]);

    await expect(readArtifact('/get-artifact')).rejects.toBeInstanceOf(ArtifactPreviewTooLargeError);

    expect(read).toHaveBeenCalledTimes(2);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('an understated Content-Length cannot bypass the byte limit', async () => {
    const { cancel, releaseLock } = mockArtifactStream([new Uint8Array(MAX_ARTIFACT_PREVIEW_BYTES + 1)], {
      'Content-Length': '1',
    });
    await expect(getArtifactBytesContent('/get-artifact')).rejects.toBeInstanceOf(ArtifactPreviewTooLargeError);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('an overstated Content-Length does not reject a small artifact', async () => {
    const { cancel, releaseLock } = mockArtifactStream([new Uint8Array([0xe2]), new Uint8Array([0x82, 0xac])], {
      'Content-Length': String(MAX_ARTIFACT_PREVIEW_BYTES + 1),
    });
    await expect(getArtifactContent('/get-artifact')).resolves.toBe('€');
    expect(cancel).not.toHaveBeenCalled();
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('allows exactly 50 MiB and preserves the original binary bytes', async () => {
    const first = new Uint8Array(MAX_ARTIFACT_PREVIEW_BYTES - 1);
    first[0] = 0xff;
    const { cancel, releaseLock } = mockArtifactStream([first, new Uint8Array([0x80])]);
    const result = new Uint8Array(await getArtifactBytesContent('/get-artifact'));
    expect(result.byteLength).toBe(MAX_ARTIFACT_PREVIEW_BYTES);
    expect(result[0]).toBe(0xff);
    expect(result[result.length - 1]).toBe(0x80);
    expect(cancel).not.toHaveBeenCalled();
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('preserves the stream error when cancellation also fails', async () => {
    const { read, cancel, releaseLock } = mockArtifactStream([]);
    const readError = new TypeError('stream failed');
    read.mockRejectedValueOnce(readError);
    cancel.mockRejectedValueOnce(new Error('cancel failed'));
    await expect(getArtifactContent('/get-artifact')).rejects.toBe(readError);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(releaseLock).toHaveBeenCalledTimes(1);
  });

  test('preserves the content type for blob previews', async () => {
    mockArtifactStream([new Uint8Array([1, 2, 3])], { 'Content-Type': 'image/svg+xml' });
    const blob = await getArtifactBlob('/get-artifact');
    expect(blob.type).toBe('image/svg+xml');
    expect(blob.size).toBe(3);
  });

  test.each([undefined, { 'x-storage-header': 'required' }])(
    'downloads remain unbounded with headers %j',
    async (headers) => {
      const blob = new Blob([new Uint8Array(MAX_ARTIFACT_PREVIEW_BYTES + 1)]);
      const readBlob = jest.fn<() => Promise<Blob>>().mockResolvedValue(blob);
      const fetchSpy = jest
        .spyOn(global, 'fetch')
        .mockResolvedValue({ ok: true, blob: readBlob } as unknown as Response);

      await expect(getArtifactDownloadBlob('https://storage.example/artifact', { headers })).resolves.toBe(blob);
      expect(readBlob).toHaveBeenCalledTimes(1);
      if (headers) {
        expect((fetchSpy.mock.calls[0][0] as Request).headers.get('x-storage-header')).toBe('required');
      }
    },
  );
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

  test('reads evaluation data larger than the preview limit', async () => {
    const bytes = new Uint8Array(MAX_ARTIFACT_PREVIEW_BYTES + 1).fill(0x61);
    const { cancel, releaseLock } = mockArtifactStream([bytes]);

    await expect(getArtifactChunkedText('artifact')).resolves.toHaveLength(bytes.length);
    expect(cancel).not.toHaveBeenCalled();
    expect(releaseLock).toHaveBeenCalledTimes(1);
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
