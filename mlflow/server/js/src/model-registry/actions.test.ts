import { afterEach, describe, test, expect, jest } from '@jest/globals';
import * as ArtifactUtils from '../common/utils/ArtifactUtils';
import * as PresignedArtifactUtils from '../experiment-tracking/utils/PresignedArtifactUtils';
import { fetchModelVersionArtifact, resolveFilterValue } from './actions';
import { Services } from './services';

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
  test('uses the run-scoped presigned flow for a run-backed model version', async () => {
    jest.spyOn(Services, 'getModelVersion').mockResolvedValue({
      model_version: { run_id: 'run-123', source: 'runs:/run-123/model' },
    });
    const fetchSpy = jest
      .spyOn(PresignedArtifactUtils, 'fetchRunArtifactWithPresignedUrl')
      .mockResolvedValue('model contents');

    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('model contents');
    expect(fetchSpy).toHaveBeenCalledWith(
      'run-123',
      'model/MLmodel',
      'model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1',
      ArtifactUtils.getArtifactContent,
    );
  });

  test('uses the logged-model presigned flow for a logged-model-backed version', async () => {
    jest.spyOn(Services, 'getModelVersion').mockResolvedValue({
      model_version: { model_id: 'm-123', source: 'models:/m-123' },
    });
    const fetchSpy = jest
      .spyOn(PresignedArtifactUtils, 'fetchArtifactWithPresignedUrl')
      .mockResolvedValue('model contents');

    await expect(fetchModelVersionArtifact('Model A', '1')).resolves.toBe('model contents');
    expect(fetchSpy).toHaveBeenCalledWith(
      {
        runUuid: '',
        path: 'MLmodel',
        isLoggedModelsMode: true,
        loggedModelId: 'm-123',
      },
      'model-versions/get-artifact?path=MLmodel&name=Model%20A&version=1',
      ArtifactUtils.getArtifactContent,
    );
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
