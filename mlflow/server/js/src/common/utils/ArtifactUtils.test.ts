import { afterEach, describe, expect, test } from '@jest/globals';

import { getArtifactLocationUrl, getLoggedModelArtifactLocationUrl } from './ArtifactUtils';
import { setActiveWorkspace } from './WorkspaceUtils';

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
