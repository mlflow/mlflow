import { afterEach, describe, expect, jest, test } from '@jest/globals';

import { getArtifactBlob, getArtifactLocationUrl, getLoggedModelArtifactLocationUrl } from './ArtifactUtils';
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
