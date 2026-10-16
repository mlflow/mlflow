import { afterEach, describe, expect, jest, test } from '@jest/globals';

import { getArtifactChunkedText, getArtifactLocationUrl, getLoggedModelArtifactLocationUrl } from './ArtifactUtils';
import { ErrorWrapper } from './ErrorWrapper';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

describe('ArtifactUtils workspace-aware URLs', () => {
  afterEach(() => {
    setActiveWorkspace(null);
  });

  test('getArtifactLocationUrl omits workspace segment and relies on headers', () => {
    setActiveWorkspace('team-a');
    const url = getArtifactLocationUrl('file.txt', 'run-123');

    expect(url).toContain('get-artifact');
    expect(url).not.toContain('workspaces');
    expect(url).toContain('path=file.txt');
    expect(url).toContain('run_uuid=run-123');
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
