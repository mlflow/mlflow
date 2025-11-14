import { afterAll, afterEach, beforeAll, describe, expect, it, jest } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { fetchArtifactUnified } from './fetchArtifactUnified';
import { setActiveWorkspace } from '../../../../common/utils/WorkspaceUtils';

describe('fetchArtifactUnified', () => {
  const experimentId = 'test-experiment-id';
  const runUuid = 'test-run-uuid';
  const isLoggedModelsMode = true;
  const loggedModelId = 'test-logged-model-id';
  const path = '/path/to/artifact';

  const runArtifactContent = 'test-run-artifact-content';
  const loggedModelArtifactContent = 'test-logged-model-artifact-content';

  const server = setupServer(
    rest.get(/\/?get-artifact/, (req, res, ctx) => {
      expect(req.headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
      return res(ctx.body(runArtifactContent));
    }),
    rest.get(/\/?ajax-api\/2\.0\/mlflow\/get-artifact/, (req, res, ctx) => {
      expect(req.headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
      return res(ctx.body(runArtifactContent));
    }),
    rest.get(/\/?ajax-api\/2\.0\/mlflow\/logged-models\/test-logged-model-id\/artifacts\/files/, (req, res, ctx) => {
      expect(req.headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
      return res(ctx.body(loggedModelArtifactContent));
    }),
  );

  beforeAll(() => {
    setActiveWorkspace('team-a');
    server.listen();
  });

  afterEach(() => {
    server.resetHandlers();
  });

  afterAll(() => {
    setActiveWorkspace(null);
    server.close();
  });

  it('fetches run artifact from workspace API', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await fetchArtifactUnified({
      experimentId,
      path,
      runUuid,
      isLoggedModelsMode: false,
      loggedModelId: undefined,
    });

    expect(result).toEqual(runArtifactContent);
    jest.restoreAllMocks();
  });

  it('fetches logged model artifact from workspace API', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await fetchArtifactUnified({ experimentId, path, runUuid, isLoggedModelsMode, loggedModelId });

    expect(result).toEqual(loggedModelArtifactContent);
    jest.restoreAllMocks();
  });
});
