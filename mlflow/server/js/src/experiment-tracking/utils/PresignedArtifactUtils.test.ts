import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import {
  getArtifactsPresignedOnlySync,
  getMultipartDownloadsEnabledSync,
  getMultipartUploadsEnabledSync,
  getPresignedUploadRunIdSupportedSync,
} from '../hooks/useServerInfo';
import { MlflowService } from '../sdk/MlflowService';
import {
  fetchArtifactWithPresignedUrl,
  getProxiedArtifactRoute,
  resolvePresignedArtifactDownload,
  uploadArtifactWithPresignedUrl,
} from './PresignedArtifactUtils';

jest.mock('../hooks/useServerInfo', () => ({
  getArtifactsPresignedOnlySync: jest.fn(),
  getMultipartDownloadsEnabledSync: jest.fn(),
  getMultipartUploadsEnabledSync: jest.fn(),
  getPresignedUploadRunIdSupportedSync: jest.fn(),
}));

const mockedPresignedOnly = jest.mocked(getArtifactsPresignedOnlySync);
const mockedMultipartDownloads = jest.mocked(getMultipartDownloadsEnabledSync);
const mockedMultipartUploads = jest.mocked(getMultipartUploadsEnabledSync);
const mockedRunUploadSupported = jest.mocked(getPresignedUploadRunIdSupportedSync);

describe('presigned artifact utilities', () => {
  beforeEach(() => {
    mockedPresignedOnly.mockReturnValue(false);
    mockedMultipartDownloads.mockReturnValue(false);
    mockedMultipartUploads.mockReturnValue(false);
    mockedRunUploadSupported.mockReturnValue(false);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('derives proxied paths and preserves an HTTP deployment prefix', () => {
    expect(getProxiedArtifactRoute('mlflow-artifacts:/12/run-id/artifacts', 'model/file.pkl')).toEqual({
      path: '12/run-id/artifacts/model/file.pkl',
    });
    expect(
      getProxiedArtifactRoute(
        'https://artifacts.example/mlflow/api/2.0/mlflow-artifacts/artifacts/12/run-id/artifacts',
        'model/file.pkl',
      ),
    ).toEqual({
      path: '12/run-id/artifacts/model/file.pkl',
      artifactServiceBaseUrl: 'https://artifacts.example/mlflow/',
    });
    expect(getProxiedArtifactRoute('s3://bucket/run-id/artifacts', 'model/file.pkl')).toBeUndefined();
  });

  it('fetches a proxied preview from object storage with only the required headers', async () => {
    mockedMultipartDownloads.mockReturnValue(true);
    jest.spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl').mockResolvedValue({
      url: 'https://storage.example/file',
      headers: { 'x-storage-header': 'value' },
    });
    const getArtifact = jest.fn(async (_url: string, _options?: unknown) => 'contents');

    const result = await fetchArtifactWithPresignedUrl(
      {
        runUuid: 'run-id',
        path: 'file.txt',
        artifactRootUri: 'mlflow-artifacts:/12/run-id/artifacts',
      },
      '/get-artifact?run_uuid=run-id&path=file.txt',
      getArtifact,
    );

    expect(result).toBe('contents');
    expect(getArtifact).toHaveBeenCalledWith('https://storage.example/file', {
      headers: { 'x-storage-header': 'value' },
    });
  });

  it('falls back for an unsupported presigned download unless the server enforces it', async () => {
    mockedMultipartDownloads.mockReturnValue(true);
    jest
      .spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl')
      .mockRejectedValue(new ErrorWrapper('unsupported', 501));
    const params = {
      runUuid: 'run-id',
      path: 'file.txt',
      artifactRootUri: 'mlflow-artifacts:/12/run-id/artifacts',
    };

    await expect(resolvePresignedArtifactDownload(params)).resolves.toBeUndefined();

    mockedPresignedOnly.mockReturnValue(true);
    await expect(resolvePresignedArtifactDownload(params)).rejects.toMatchObject({ status: 501 });
  });

  it('falls back when a browser CORS error blocks a presigned preview unless the server enforces it', async () => {
    mockedMultipartDownloads.mockReturnValue(true);
    jest.spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl').mockResolvedValue({
      url: 'https://storage.example/file',
      headers: {},
    });
    const getArtifact = jest
      .fn<(url: string, options?: unknown) => Promise<string>>()
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce('legacy contents');
    const params = {
      runUuid: 'run-id',
      path: 'file.txt',
      artifactRootUri: 'mlflow-artifacts:/12/run-id/artifacts',
    };

    await expect(fetchArtifactWithPresignedUrl(params, '/get-artifact', getArtifact)).resolves.toBe('legacy contents');
    expect(getArtifact).toHaveBeenNthCalledWith(1, 'https://storage.example/file', { headers: {} });
    expect(getArtifact).toHaveBeenNthCalledWith(2, '/get-artifact');

    mockedPresignedOnly.mockReturnValue(true);
    getArtifact.mockReset().mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(fetchArtifactWithPresignedUrl(params, '/get-artifact', getArtifact)).rejects.toThrow(
      'Failed to fetch',
    );
    expect(getArtifact).toHaveBeenCalledTimes(1);
  });

  it('resolves direct logged-model artifacts through their download credentials', async () => {
    jest.spyOn(MlflowService, 'getCredentialsForLoggedModelArtifactRead').mockResolvedValue({
      credentials: [
        {
          credential_info: {
            type: 'AWS_PRESIGNED_URL',
            signed_uri: 'https://storage.example/model-file',
            path: 'MLmodel',
            headers: [{ name: 'x-storage-header', value: 'required' }],
          },
        },
      ],
    });

    await expect(
      resolvePresignedArtifactDownload({
        runUuid: '',
        path: 'MLmodel',
        artifactRootUri: 's3://bucket/logged-models/model-id/artifacts',
        isLoggedModelsMode: true,
        loggedModelId: 'model-id',
      }),
    ).resolves.toEqual({
      url: 'https://storage.example/model-file',
      headers: { 'x-storage-header': 'required' },
    });
  });

  it('resolves a logged-model artifact without requiring its root URI', async () => {
    const getRunSpy = jest.spyOn(MlflowService, 'getRun');
    jest.spyOn(MlflowService, 'getCredentialsForLoggedModelArtifactRead').mockResolvedValue({
      credentials: [
        {
          credential_info: {
            type: 'AWS_PRESIGNED_URL',
            signed_uri: 'https://storage.example/model-file',
            path: 'MLmodel',
          },
        },
      ],
    });

    await expect(
      resolvePresignedArtifactDownload({
        runUuid: '',
        path: 'MLmodel',
        isLoggedModelsMode: true,
        loggedModelId: 'model-id',
      }),
    ).resolves.toEqual({
      url: 'https://storage.example/model-file',
      headers: {},
    });
    expect(getRunSpy).not.toHaveBeenCalled();
  });

  it('uploads a direct run artifact through its presigned URL', async () => {
    mockedRunUploadSupported.mockReturnValue(true);
    jest.spyOn(MlflowService, 'getRun').mockResolvedValue({
      run: { info: { artifactUri: 's3://bucket/run-id/artifacts' } },
    } as any);
    jest.spyOn(MlflowService, 'createPresignedUploadUrl').mockResolvedValue({
      presigned_url: 'https://storage.example/upload',
      headers: { 'Content-Type': 'application/json' },
    });
    const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue(new Response(undefined, { status: 200 }));

    await expect(uploadArtifactWithPresignedUrl('run-id', 'prompt.json', '{"value":1}')).resolves.toBe(true);
    expect(fetchSpy).toHaveBeenCalledWith('https://storage.example/upload', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: '{"value":1}',
    });
  });

  it('allows a legacy upload fallback after a browser CORS error unless the server enforces it', async () => {
    mockedRunUploadSupported.mockReturnValue(true);
    jest.spyOn(MlflowService, 'getRun').mockResolvedValue({
      run: { info: { artifactUri: 's3://bucket/run-id/artifacts' } },
    } as any);
    jest.spyOn(MlflowService, 'createPresignedUploadUrl').mockResolvedValue({
      presigned_url: 'https://storage.example/upload',
      headers: {},
    });
    jest.spyOn(global, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));

    await expect(uploadArtifactWithPresignedUrl('run-id', 'prompt.json', '{}')).resolves.toBe(false);

    mockedPresignedOnly.mockReturnValue(true);
    await expect(uploadArtifactWithPresignedUrl('run-id', 'prompt.json', '{}')).rejects.toThrow('Failed to fetch');
  });

  it('completes a one-part upload for proxied run artifacts', async () => {
    mockedMultipartUploads.mockReturnValue(true);
    jest.spyOn(MlflowService, 'getRun').mockResolvedValue({
      run: { info: { artifactUri: 'mlflow-artifacts:/12/run-id/artifacts' } },
    } as any);
    const createSpy = jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload').mockResolvedValue({
      upload_id: 'upload-id',
      credentials: [{ url: 'https://storage.example/part', part_number: 1, headers: {} }],
    });
    const completeSpy = jest.spyOn(MlflowService, 'completeMlflowArtifactsMultipartUpload').mockResolvedValue({});
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(undefined, {
        status: 200,
        headers: { ETag: 'part-etag' },
      }),
    );

    await expect(uploadArtifactWithPresignedUrl('run-id', 'evaluations/table.json', '{}')).resolves.toBe(true);
    expect(createSpy).toHaveBeenCalledWith('12/run-id/artifacts/evaluations', { path: 'table.json', num_parts: 1 });
    expect(completeSpy).toHaveBeenCalledWith('12/run-id/artifacts/evaluations', {
      path: 'table.json',
      upload_id: 'upload-id',
      parts: [
        {
          part_number: 1,
          etag: 'part-etag',
          url: 'https://storage.example/part',
        },
      ],
    });
  });
});
