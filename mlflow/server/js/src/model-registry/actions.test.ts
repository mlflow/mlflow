import { afterEach, describe, test, expect, jest } from '@jest/globals';
import * as ArtifactUtils from '../common/utils/ArtifactUtils';
import * as PresignedArtifactUtils from '../experiment-tracking/utils/PresignedArtifactUtils';
import { fetchModelVersionArtifact, resolveFilterValue } from './actions';
import { Services } from './services';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import { ErrorWrapper } from '../common/utils/ErrorWrapper';
import * as ServerInfo from '../experiment-tracking/hooks/useServerInfo';

afterEach(() => {
  jest.restoreAllMocks();
});

describe('action tests', () => {
  test('simple string', () => {
    expect(resolveFilterValue('hello')).toEqual("'hello'");
  });

  test('simple string with wildcard', () => {
    expect(resolveFilterValue('hello', true)).toEqual("'%hello%'");
  });

  test('simple string spaces', () => {
    expect(resolveFilterValue(' he llo  ')).toEqual("' he llo  '");
  });

  test('simple string spaces with wildcard', () => {
    expect(resolveFilterValue(' he llo  ', true)).toEqual("'% he llo  %'");
  });

  test('single quotes', () => {
    expect(resolveFilterValue("A's model")).toEqual('"A\'s model"');
  });

  test('single quotes with wildcard', () => {
    expect(resolveFilterValue("A's model", true)).toEqual('"%A\'s model%"');
  });

  test('double quotes', () => {
    expect(resolveFilterValue('the "best" model')).toEqual('\'the "best" model\'');
  });

  test('double quotes with wildcard', () => {
    expect(resolveFilterValue('the "best" model', true)).toEqual('\'%the "best" model%\'');
  });

  test('percent character', () => {
    expect(resolveFilterValue('%')).toEqual("'%'");
  });

  test('percent character with wildcard', () => {
    expect(resolveFilterValue('%', true)).toEqual("'%%%'");
  });
});

describe('fetchModelVersionArtifact', () => {
  test('prompts an open tab to reload after the server starts requiring presigned transfers', async () => {
    jest.spyOn(ServerInfo, 'getArtifactsPresignedOnlySync').mockReturnValue(false);
    jest.spyOn(Services, 'getModelVersion').mockResolvedValue({
      model_version: { run_id: 'run-123', source: 'runs:/run-123/model' },
    });
    const sourceRun = jest.spyOn(MlflowService, 'getRun').mockRejectedValue(new ErrorWrapper('source denied', 403));
    const rejection = new ErrorWrapper(
      {
        error_code: 'PERMISSION_DENIED',
        message: 'This server requires artifacts to be downloaded directly from storage.',
        artifacts_presigned_only: true,
      },
      403,
    );
    const registryRead = jest.spyOn(ArtifactUtils, 'getArtifactContent').mockRejectedValue(rejection);
    const sign = jest.spyOn(MlflowService, 'getModelVersionPresignedDownloadUrl');
    await expect(fetchModelVersionArtifact('Model A', '1')).rejects.toBe(rejection);
    expect(rejection.renderHttpError()).toContain('Reload this page');
    expect(sourceRun).toHaveBeenCalledTimes(1);
    expect(registryRead).toHaveBeenCalledTimes(1);
    expect(sign).not.toHaveBeenCalled();
  });

  test('uses the run-scoped presigned flow for a run-backed model version', async () => {
    jest.spyOn(Services, 'getModelVersion').mockResolvedValue({
      model_version: { run_id: 'run-123', source: 'runs:/run-123/model' },
    });
    jest
      .spyOn(MlflowService, 'getRun')
      .mockResolvedValue({ run: { info: { artifactUri: 'mlflow-artifacts:/12/run-123/artifacts' } } } as any);
    const fetchSpy = jest
      .spyOn(PresignedArtifactUtils, 'fetchArtifactPreferPresigned')
      .mockResolvedValue('model contents');

    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('model contents');
    expect(fetchSpy).toHaveBeenCalledWith(
      { runUuid: 'run-123', path: 'model/MLmodel', artifactRootUri: 'mlflow-artifacts:/12/run-123/artifacts' },
      'model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1',
      ArtifactUtils.getArtifactContent,
    );
  });

  test('uses the logged-model presigned flow for a logged-model-backed version', async () => {
    jest.spyOn(Services, 'getModelVersion').mockResolvedValue({
      model_version: { model_id: 'm-123', source: 'models:/m-123' },
    });
    jest
      .spyOn(MlflowService, 'getLoggedModel')
      .mockResolvedValue({ model: { info: { artifact_uri: 'mlflow-artifacts:/12/models/m-123/artifacts' } } });
    const fetchSpy = jest
      .spyOn(PresignedArtifactUtils, 'fetchArtifactPreferPresigned')
      .mockResolvedValue('model contents');

    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('model contents');
    expect(fetchSpy).toHaveBeenCalledWith(
      {
        runUuid: '',
        path: 'MLmodel',
        isLoggedModelsMode: true,
        loggedModelId: 'm-123',
        artifactRootUri: 'mlflow-artifacts:/12/models/m-123/artifacts',
      },
      'model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1',
      ArtifactUtils.getArtifactContent,
    );
  });

  test('preserves registry authorization when source-run access is denied and proxying is allowed', async () => {
    jest.spyOn(ServerInfo, 'getArtifactsPresignedOnlySync').mockReturnValue(false);
    jest
      .spyOn(Services, 'getModelVersion')
      .mockResolvedValue({ model_version: { run_id: 'run-123', source: 'runs:/run-123/model' } });
    jest.spyOn(MlflowService, 'getRun').mockRejectedValue(new ErrorWrapper('source access denied', 403));
    const legacy = jest.spyOn(ArtifactUtils, 'getArtifactContent').mockResolvedValue('registry contents');
    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('registry contents');
    expect(legacy).toHaveBeenCalledWith('model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1');
  });

  test('uses model-version authorization in presigned-only mode without looking up the source', async () => {
    jest.spyOn(ServerInfo, 'getArtifactsPresignedOnlySync').mockReturnValue(true);
    const sourceRun = jest.spyOn(MlflowService, 'getRun');
    const modelVersion = jest.spyOn(Services, 'getModelVersion');
    const sign = jest
      .spyOn(MlflowService, 'getModelVersionPresignedDownloadUrl')
      .mockResolvedValue({ url: 'https://storage.example/MLmodel', headers: { 'x-storage': 'required' } });
    const fetchArtifact = jest
      .spyOn(PresignedArtifactUtils, 'fetchArtifactPreferPresigned')
      .mockResolvedValue('registry contents');
    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('registry contents');
    expect(sign).toHaveBeenCalledWith({ name: 'Model A', version: '1', path: 'MLmodel' });
    expect(fetchArtifact).toHaveBeenCalledWith(
      { runUuid: '', path: 'MLmodel' },
      'model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1',
      ArtifactUtils.getArtifactContent,
      { url: 'https://storage.example/MLmodel', headers: { 'x-storage': 'required' } },
    );
    expect(sourceRun).not.toHaveBeenCalled();
    expect(modelVersion).not.toHaveBeenCalled();
  });

  test('reads registered model artifacts after the source logged model is soft-deleted', async () => {
    jest.spyOn(ServerInfo, 'getArtifactsPresignedOnlySync').mockReturnValue(false);
    const version = jest.spyOn(Services, 'getModelVersion').mockResolvedValue({
      model_version: { model_id: 'm-deleted', source: 'models:/m-deleted' },
    });
    const model = jest.spyOn(MlflowService, 'getLoggedModel').mockRejectedValue(new ErrorWrapper('model deleted', 404));
    const registryRead = jest.spyOn(ArtifactUtils, 'getArtifactContent').mockResolvedValue('registry contents');
    const presignedRead = jest.spyOn(PresignedArtifactUtils, 'fetchArtifactPreferPresigned');

    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('registry contents');
    expect(version).toHaveBeenCalledWith({ name: 'Model A', version: '1' });
    expect(model).toHaveBeenCalledWith('m-deleted');
    expect(registryRead).toHaveBeenCalledWith('model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1');
    expect(presignedRead).not.toHaveBeenCalled();
  });

  test('never falls back after model-version signing is forbidden', async () => {
    jest.spyOn(ServerInfo, 'getArtifactsPresignedOnlySync').mockReturnValue(true);
    const error = new ErrorWrapper('version denied', 403);
    jest.spyOn(MlflowService, 'getModelVersionPresignedDownloadUrl').mockRejectedValue(error);
    const legacy = jest.spyOn(ArtifactUtils, 'getArtifactContent');
    await expect(fetchModelVersionArtifact('Model A', '1')).rejects.toBe(error);
    expect(legacy).not.toHaveBeenCalled();
  });

  test.each([403, 404])('does not swallow a %s failure from the presigned transfer itself', async (status) => {
    jest
      .spyOn(Services, 'getModelVersion')
      .mockResolvedValue({ model_version: { run_id: 'run-123', source: 'runs:/run-123/model' } });
    jest.spyOn(MlflowService, 'getRun').mockResolvedValue({ run: { info: { artifactUri: 's3://bucket/run' } } } as any);
    const error = new ErrorWrapper('storage failed', status);
    jest.spyOn(PresignedArtifactUtils, 'fetchArtifactPreferPresigned').mockRejectedValue(error);
    const legacy = jest.spyOn(ArtifactUtils, 'getArtifactContent');
    await expect(fetchModelVersionArtifact('Model A', '1')).rejects.toBe(error);
    expect(legacy).not.toHaveBeenCalled();
  });

  test('does not attempt a proxy read for an unresolved source in presigned-only mode', async () => {
    jest.spyOn(ServerInfo, 'getArtifactsPresignedOnlySync').mockReturnValue(true);
    jest.spyOn(MlflowService, 'getModelVersionPresignedDownloadUrl').mockResolvedValue({});
    const legacy = jest.spyOn(ArtifactUtils, 'getArtifactContent');
    await expect(fetchModelVersionArtifact('Model A', '1')).rejects.toMatchObject({ status: 501 });
    expect(legacy).not.toHaveBeenCalled();
  });

  test('retains the legacy route for model versions without a resolvable source identity', async () => {
    jest.spyOn(Services, 'getModelVersion').mockResolvedValue({
      model_version: { source: 's3://external-bucket/model' },
    });
    const legacyFetchSpy = jest.spyOn(ArtifactUtils, 'getArtifactContent').mockResolvedValue('model contents');

    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('model contents');
    expect(legacyFetchSpy).toHaveBeenCalledWith('model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1');
  });
});
