import { afterAll, afterEach, beforeAll, describe, expect, it, jest } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { fetchArtifactUnified } from './fetchArtifactUnified';
import { setActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../../hooks/useServerInfo';

jest.mock('../../../hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../../hooks/useServerInfo')>('../../../hooks/useServerInfo'),
  getWorkspacesEnabledSync: jest.fn(),
}));

const getWorkspacesEnabledSyncMock = jest.mocked(getWorkspacesEnabledSync);

describe('fetchArtifactUnified', () => {
  const experimentId = 'test-experiment-id';
  const runUuid = 'test-run-uuid';
  const isLoggedModelsMode = true;
  const loggedModelId = 'test-logged-model-id';
  const path = '/path/to/artifact';

  const runArtifactContent = 'test-run-artifact-content';
  const loggedModelArtifactContent = 'test-logged-model-artifact-content';

  let capturedRequests: { url: string; workspaceHeader: string | null }[] = [];
  const server = setupServer(
    rest.get(/\/?get-artifact/, (req, res, ctx) => {
      capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
      return res(ctx.body(runArtifactContent));
    }),
    rest.get(/\/?ajax-api\/2\.0\/mlflow\/get-artifact/, (req, res, ctx) => {
      capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
      return res(ctx.body(runArtifactContent));
    }),
    rest.get(/\/?ajax-api\/2\.0\/mlflow\/logged-models\/test-logged-model-id\/artifacts\/files/, (req, res, ctx) => {
      capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
      return res(ctx.body(loggedModelArtifactContent));
    }),
  );

  beforeAll(() => {
    getWorkspacesEnabledSyncMock.mockReturnValue(true);
    setActiveWorkspace('team-a');
    server.listen();
  });

  afterEach(() => {
    capturedRequests = [];
    server.resetHandlers();
  });

  afterAll(() => {
    setActiveWorkspace(null);
    jest.restoreAllMocks();
    server.close();
  });

  it('fetches run artifact from workspace API', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await fetchArtifactUnified({
      experimentId,
      path,
      runUuid,
      isLoggedModelsMode: false,
      loggedModelId: undefined,
    });

    expect(result).toEqual(runArtifactContent);

    expect(capturedRequests.length).toBeGreaterThan(0);
    for (const req of capturedRequests) {
      expect(req.workspaceHeader).toBe('team-a');
    }
  });

  it('fetches logged model artifact from workspace API', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await fetchArtifactUnified({ experimentId, path, runUuid, isLoggedModelsMode, loggedModelId });

    expect(result).toEqual(loggedModelArtifactContent);

    expect(capturedRequests.length).toBeGreaterThan(0);
    for (const req of capturedRequests) {
      expect(req.workspaceHeader).toBe('team-a');
    }
  });

  it('fetches run artifact directly from an eligible artifact proxy URI', async () => {
    const proxyArtifactContent = 'test-proxy-artifact-content';
    server.use(
      rest.get(/\/api\/2\.0\/mlflow-artifacts\/artifacts\/my-root\/path\/to\/artifact/, (req, res, ctx) => {
        capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
        return res(ctx.body(proxyArtifactContent));
      }),
    );

    const result = await fetchArtifactUnified({
      experimentId,
      path,
      runUuid,
      isLoggedModelsMode: false,
      loggedModelId: undefined,
      artifactUri: 'http://localhost/api/2.0/mlflow-artifacts/artifacts/my-root',
    });

    expect(result).toEqual(proxyArtifactContent);
    expect(capturedRequests).toHaveLength(1);
    expect(capturedRequests[0].url).toBe(
      'http://localhost/api/2.0/mlflow-artifacts/artifacts/my-root/path/to/artifact',
    );
    // The issue requires the UI's existing headers to reach the artifact server.
    expect(capturedRequests[0].workspaceHeader).toBe('team-a');
  });

  it('falls back to the tracking server when the artifact URI is not eligible', async () => {
    const result = await fetchArtifactUnified({
      experimentId,
      path,
      runUuid,
      isLoggedModelsMode: false,
      loggedModelId: undefined,
      artifactUri: 'http://artifacts.example.com/api/2.0/mlflow-artifacts/artifacts/my-root',
    });

    expect(result).toEqual(runArtifactContent);
    expect(capturedRequests).toHaveLength(1);
    expect(capturedRequests[0].url).toContain('get-artifact');
  });

  it('uses the logged model endpoint when only the run artifact root is known', async () => {
    // In logged models mode the artifact root belongs to the run, not the model,
    // so routing to it would fetch from the wrong artifact root.
    const result = await fetchArtifactUnified({
      experimentId,
      path,
      runUuid,
      isLoggedModelsMode,
      loggedModelId,
      artifactUri: 'http://localhost/api/2.0/mlflow-artifacts/artifacts/run-root',
      isArtifactUriForEntity: false,
    });

    expect(result).toEqual(loggedModelArtifactContent);
    expect(capturedRequests[0].url).toContain('logged-models');
  });

  it('fetches logged model artifact directly from an eligible artifact proxy URI', async () => {
    const proxyArtifactContent = 'test-proxy-logged-model-content';
    server.use(
      rest.get(/\/api\/2\.0\/mlflow-artifacts\/artifacts\/model-root\/path\/to\/artifact/, (req, res, ctx) => {
        capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
        return res(ctx.body(proxyArtifactContent));
      }),
    );

    const result = await fetchArtifactUnified({
      experimentId,
      path,
      runUuid,
      isLoggedModelsMode,
      loggedModelId,
      artifactUri: 'http://localhost/api/2.0/mlflow-artifacts/artifacts/model-root',
      isArtifactUriForEntity: true,
    });

    expect(result).toEqual(proxyArtifactContent);
    expect(capturedRequests).toHaveLength(1);
  });
});
