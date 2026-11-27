import { afterAll, afterEach, beforeAll, describe, expect, it, jest } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { fetchArtifactUnified } from './fetchArtifactUnified';
import { setActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../../hooks/useServerInfo';
import { getDefaultHeaders } from '../../../../common/utils/FetchUtils';

jest.mock('../../../hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../../hooks/useServerInfo')>('../../../hooks/useServerInfo'),
  getWorkspacesEnabledSync: jest.fn(),
}));

const getWorkspacesEnabledSyncMock = jest.mocked(getWorkspacesEnabledSync);

describe('fetchArtifactUnified', () => {
  // Jest's fetch polyfill has no ReadableStream. These tests cover route/workspace selection;
  // ArtifactUtils tests exercise the bounded streaming readers with browser-shaped responses.
  const readArtifactText = async (url: string) => {
    // eslint-disable-next-line no-restricted-globals -- Test-only callback for the fetch polyfill.
    const response = await fetch(new Request(url, { headers: getDefaultHeaders(document.cookie) as HeadersInit }));
    return response.text();
  };
  const runUuid = 'test-run-uuid';
  const isLoggedModelsMode = true;
  const loggedModelId = 'test-logged-model-id';
  const path = '/path/to/artifact';

  const runArtifactContent = 'test-run-artifact-content';
  const loggedModelArtifactContent = 'test-logged-model-artifact-content';

  let capturedRequests: { url: string; workspaceHeader: string | null }[] = [];
  const server = setupServer(
    rest.get(/\/ajax-api\/2\.0\/mlflow\/runs\/get$/, (req, res, ctx) => {
      capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
      return res(ctx.json({ run: { info: { artifact_uri: 'mlflow-artifacts:/1/test-run-uuid/artifacts' } } }));
    }),
    rest.get(/\/logged-models\/test-logged-model-id$/, (req, res, ctx) => {
      capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
      return res(ctx.json({ model: { info: { artifact_uri: 's3://bucket/model/artifacts' } } }));
    }),
    rest.post(/\/logged-models\/test-logged-model-id\/artifacts\/credentials-for-download/, (req, res, ctx) => {
      capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
      return res(ctx.status(501));
    }),
    rest.get(
      /\/ajax-api\/2\.0\/mlflow-artifacts\/artifacts\/1\/test-run-uuid\/artifacts\/path\/to\/artifact$/,
      (req, res, ctx) => {
        capturedRequests.push({ url: req.url.toString(), workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE') });
        return res(ctx.body(runArtifactContent));
      },
    ),
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
    const result = await fetchArtifactUnified(
      {
        path,
        runUuid,
        isLoggedModelsMode: false,
        loggedModelId: undefined,
      },
      readArtifactText,
    );

    expect(result).toEqual(runArtifactContent);
    expect(capturedRequests.map(({ url }) => new URL(url).pathname)).toEqual([
      '/ajax-api/2.0/mlflow/runs/get',
      '/ajax-api/2.0/mlflow-artifacts/artifacts/1/test-run-uuid/artifacts/path/to/artifact',
    ]);
    for (const req of capturedRequests) {
      expect(req.workspaceHeader).toBe('team-a');
    }
  });

  it('fetches logged model artifact from workspace API', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await fetchArtifactUnified({ path, runUuid, isLoggedModelsMode, loggedModelId }, readArtifactText);

    expect(result).toEqual(loggedModelArtifactContent);

    expect(capturedRequests.length).toBeGreaterThan(0);
    for (const req of capturedRequests) {
      expect(req.workspaceHeader).toBe('team-a');
    }
  });
});
