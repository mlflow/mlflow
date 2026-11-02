import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { getDefaultHeaders } from '../../common/utils/FetchUtils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { uploadArtifactApi } from '../actions';
import {
  getArtifactsPresignedOnlySync,
  getLoggedModelArtifactReadCredentialsSupportedSync,
  getModelArtifactPresignedDownloadsSupportedSync,
  getMultipartDownloadsEnabledSync,
  getMultipartUploadsEnabledSync,
  getPresignedUploadRunIdSupportedSync,
} from '../hooks/useServerInfo';
import { MlflowService } from '../sdk/MlflowService';
import {
  fetchArtifactPreferPresigned,
  fetchRunArtifactPreferPresigned,
  getProxiedArtifactRoute,
  resolveArtifactDownload,
  type PresignedArtifactParams,
  uploadArtifactWithPresignedUrl,
} from './PresignedArtifactUtils';

jest.mock('../hooks/useServerInfo', () => ({
  getArtifactsPresignedOnlySync: jest.fn(),
  getLoggedModelArtifactReadCredentialsSupportedSync: jest.fn(),
  getModelArtifactPresignedDownloadsSupportedSync: jest.fn(),
  getMultipartDownloadsEnabledSync: jest.fn(),
  getMultipartUploadsEnabledSync: jest.fn(),
  getPresignedUploadRunIdSupportedSync: jest.fn(),
}));

const mockedPresignedOnly = jest.mocked(getArtifactsPresignedOnlySync);
const mockedLoggedModelCredentialsSupported = jest.mocked(getLoggedModelArtifactReadCredentialsSupportedSync);
const mockedModelPresignedDownloadsSupported = jest.mocked(getModelArtifactPresignedDownloadsSupportedSync);
const mockedMultipartDownloads = jest.mocked(getMultipartDownloadsEnabledSync);
const mockedMultipartUploads = jest.mocked(getMultipartUploadsEnabledSync);
const mockedRunUploadSupported = jest.mocked(getPresignedUploadRunIdSupportedSync);

// Existing signing assertions focus on the signed response; destination tests
// below exercise the complete resolution and its compatibility URL.
const resolvePresignedArtifactDownload = async (params: PresignedArtifactParams) =>
  (await resolveArtifactDownload(params)).presigned;

describe('presigned artifact utilities', () => {
  let now = 0;
  beforeEach(() => {
    now += 60_001;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    setActiveWorkspace(null);
    mockedPresignedOnly.mockReturnValue(false);
    mockedLoggedModelCredentialsSupported.mockReturnValue(false);
    mockedModelPresignedDownloadsSupported.mockReturnValue(false);
    mockedMultipartDownloads.mockReturnValue(false);
    mockedMultipartUploads.mockReturnValue(false);
    mockedRunUploadSupported.mockReturnValue(false);
    jest.spyOn(MlflowService, 'getLoggedModel').mockResolvedValue({
      model: { info: { artifact_uri: 's3://bucket/model/artifacts' } },
    });
  });

  afterEach(() => {
    setActiveWorkspace(null);
    document.cookie = 'mlflow-request-header-Authorization=; max-age=0; path=/';
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

    const result = await fetchArtifactPreferPresigned(
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

  it.each(['http:', 'https:'])('preserves an explicit artifact authority and the %s deployment prefix', (protocol) => {
    const base = document.createElement('base');
    base.href = `${protocol}//tracking.example/nested/mlflow/`;
    document.head.append(base);
    try {
      expect(getProxiedArtifactRoute('mlflow-artifacts://artifacts.example:9000/12/run/artifacts', 'file.txt')).toEqual(
        {
          path: '12/run/artifacts/file.txt',
          artifactServiceBaseUrl: `${protocol}//artifacts.example:9000/nested/mlflow/`,
        },
      );
    } finally {
      base.remove();
    }
  });

  it.each(['http:', 'https:'])('clears the tracking port for a portless artifact authority over %s', (protocol) => {
    const base = document.createElement('base');
    base.href = `${protocol}//tracking.example:5000/nested/mlflow/`;
    document.head.append(base);
    try {
      expect(getProxiedArtifactRoute('mlflow-artifacts://artifacts.example/12/run/artifacts', 'file.txt')).toEqual({
        path: '12/run/artifacts/file.txt',
        artifactServiceBaseUrl: `${protocol}//artifacts.example/nested/mlflow/`,
      });
    } finally {
      base.remove();
    }
  });

  it.each([
    `mlflow-artifacts://${window.location.host}/12/run/artifacts`,
    `${window.location.origin}/prefix/api/2.0/mlflow-artifacts/artifacts/12/run/artifacts`,
  ])('uses an explicit same-origin artifact service with local capabilities disabled: %s', async (artifactUri) => {
    const base = document.createElement('base');
    base.href = `${window.location.origin}/prefix/`;
    document.head.append(base);
    try {
      const serviceBaseUrl = `${window.location.origin}/prefix/`;
      const sign = jest.spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl').mockResolvedValue({
        url: 'https://storage.example/download',
      });
      await expect(
        resolvePresignedArtifactDownload({ runUuid: 'run', path: 'file.txt', artifactRootUri: artifactUri }),
      ).resolves.toEqual({ url: 'https://storage.example/download', headers: {} });
      expect(sign).toHaveBeenCalledWith('12/run/artifacts/file.txt', serviceBaseUrl);

      const getRun = jest.spyOn(MlflowService, 'getRun').mockResolvedValue({ run: { info: { artifactUri } } } as any);
      const directSign = jest.spyOn(MlflowService, 'createPresignedUploadUrl');
      const create = jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload').mockResolvedValue({
        upload_id: 'upload-id',
        credentials: [{ url: 'https://storage.example/part', part_number: 1, headers: { 'x-storage': 'required' } }],
      });
      const complete = jest.spyOn(MlflowService, 'completeMlflowArtifactsMultipartUpload').mockResolvedValue({});
      const fetchPart = jest
        .spyOn(global, 'fetch')
        .mockResolvedValue(new Response(undefined, { headers: { ETag: 'part-etag' } }));
      await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'contents')).resolves.toBe(true);
      expect(getRun).toHaveBeenCalledWith({ run_id: 'run' });
      expect(create).toHaveBeenCalledWith('12/run/artifacts', { path: 'file.txt', num_parts: 1 }, serviceBaseUrl);
      expect(complete).toHaveBeenCalledWith(
        '12/run/artifacts',
        {
          path: 'file.txt',
          upload_id: 'upload-id',
          parts: [{ part_number: 1, etag: 'part-etag', url: 'https://storage.example/part' }],
        },
        serviceBaseUrl,
      );
      expect(fetchPart).toHaveBeenCalledWith('https://storage.example/part', {
        method: 'PUT',
        headers: { 'x-storage': 'required' },
        body: 'contents',
      });
      expect(directSign).not.toHaveBeenCalled();
    } finally {
      base.remove();
    }
  });

  it.each([false, true])(
    'rejects cross-origin artifact transfers without fallback (presigned-only: %s)',
    async (enforced) => {
      mockedPresignedOnly.mockReturnValue(enforced);
      const getArtifact = jest.fn(async (_url: string) => 'contents');
      const fetch = jest.spyOn(global, 'fetch');
      const getRun = jest.spyOn(MlflowService, 'getRun');
      const artifactRootUri = 'https://artifacts.example/api/2.0/mlflow-artifacts/artifacts/12/run/artifacts';
      await expect(
        fetchArtifactPreferPresigned(
          { runUuid: 'run', path: 'file.txt', artifactRootUri },
          '/get-artifact',
          getArtifact,
        ),
      ).rejects.toMatchObject({ status: 403, textJson: { external_artifact_service: true } });
      await expect(
        uploadArtifactWithPresignedUrl('run', 'file.txt', 'contents', artifactRootUri),
      ).rejects.toMatchObject({
        status: 403,
        textJson: { external_artifact_service: true },
      });
      expect(getArtifact).not.toHaveBeenCalled();
      expect(fetch).not.toHaveBeenCalled();
      expect(getRun).not.toHaveBeenCalled();
    },
  );

  it.each([
    new ErrorWrapper('run not found', 404),
    new ErrorWrapper('run denied', 403),
    new TypeError('metadata transport failed'),
  ])('does not guess an upload destination after root discovery fails: %s', async (error) => {
    const getRun = jest.spyOn(MlflowService, 'getRun').mockRejectedValue(error);
    const create = jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload');
    const proxy = jest.spyOn(MlflowService, 'uploadMlflowArtifact');
    await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'data')).rejects.toBe(error);
    mockedPresignedOnly.mockReturnValue(true);
    await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'data')).rejects.toBe(error);
    expect(getRun).toHaveBeenCalledTimes(2);
    expect(create).not.toHaveBeenCalled();
    expect(proxy).not.toHaveBeenCalled();
  });

  it('matches AJAX routes at a boundary and uses the first artifact-service route', () => {
    expect(
      getProxiedArtifactRoute(
        'https://artifacts.example/prefix/ajax-api/2.0/mlflow-artifacts/artifacts/12/api/2.0/mlflow-artifacts/artifacts/run',
        'file.txt',
      ),
    ).toEqual({
      path: '12/api/2.0/mlflow-artifacts/artifacts/run/file.txt',
      artifactServiceBaseUrl: 'https://artifacts.example/prefix/',
    });
    expect(
      getProxiedArtifactRoute('https://artifacts.example/fake-api/2.0/mlflow-artifacts/artifacts/run'),
    ).toBeUndefined();
  });

  it.each([
    { error: new ErrorWrapper(new TypeError('CORS blocked'), 500), fallback: true },
    { error: new ErrorWrapper('server failed', 500), fallback: false },
    { error: new ErrorWrapper('permission denied', 403), fallback: false },
  ])('applies fallback only to transport failures: $error', async ({ error, fallback }) => {
    mockedMultipartDownloads.mockReturnValue(true);
    jest.spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl').mockRejectedValue(error);
    const params = { runUuid: 'run-id', path: 'file.txt', artifactRootUri: 'mlflow-artifacts:/12/run/artifacts' };
    if (fallback) await expect(resolvePresignedArtifactDownload(params)).resolves.toBeUndefined();
    else await expect(resolvePresignedArtifactDownload(params)).rejects.toBe(error);
    mockedPresignedOnly.mockReturnValue(true);
    await expect(resolvePresignedArtifactDownload(params)).rejects.toBe(error);
  });

  it('falls back from wrapped upload transport failures only when proxying is allowed', async () => {
    mockedMultipartUploads.mockReturnValue(true);
    jest
      .spyOn(MlflowService, 'getRun')
      .mockResolvedValue({ run: { info: { artifactUri: 'mlflow-artifacts:/12/run/artifacts' } } } as any);
    const error = new ErrorWrapper(new TypeError('CORS blocked'), 500);
    jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload').mockRejectedValue(error);
    const proxyUpload = jest.spyOn(MlflowService, 'uploadMlflowArtifact').mockResolvedValue({});
    await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'data')).resolves.toBe(true);
    expect(proxyUpload).toHaveBeenCalledWith('12/run/artifacts/file.txt', 'data', undefined);
    mockedPresignedOnly.mockReturnValue(true);
    await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'data')).rejects.toBe(error);
    expect(proxyUpload).toHaveBeenCalledTimes(1);
  });

  it('discovers a logged model root and signs its artifact-service path in presigned-only mode', async () => {
    mockedPresignedOnly.mockReturnValue(true);
    jest
      .mocked(MlflowService.getLoggedModel)
      .mockResolvedValue({ model: { info: { artifact_uri: 'mlflow-artifacts:/12/models/m-123/artifacts' } } });
    const sign = jest
      .spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl')
      .mockResolvedValue({ url: 'https://storage.example/model' });
    const credentials = jest.spyOn(MlflowService, 'getCredentialsForLoggedModelArtifactRead');
    await expect(
      resolvePresignedArtifactDownload({
        runUuid: 'source-run',
        path: 'MLmodel',
        isLoggedModelsMode: true,
        loggedModelId: 'm-123',
      }),
    ).resolves.toEqual({ url: 'https://storage.example/model', headers: {} });
    expect(MlflowService.getLoggedModel).toHaveBeenCalledWith('m-123');
    expect(sign).toHaveBeenCalledWith('12/models/m-123/artifacts/MLmodel');
    expect(credentials).not.toHaveBeenCalled();
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

    await expect(fetchArtifactPreferPresigned(params, '/get-artifact', getArtifact)).resolves.toBe('legacy contents');
    expect(getArtifact).toHaveBeenNthCalledWith(1, 'https://storage.example/file', { headers: {} });
    expect(getArtifact).toHaveBeenNthCalledWith(
      2,
      'ajax-api/2.0/mlflow-artifacts/artifacts/12/run-id/artifacts/file.txt',
    );

    mockedPresignedOnly.mockReturnValue(true);
    getArtifact.mockReset().mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(fetchArtifactPreferPresigned(params, '/get-artifact', getArtifact)).rejects.toThrow('Failed to fetch');
    expect(getArtifact).toHaveBeenCalledTimes(1);
  });

  it('resolves direct logged-model artifacts through their download credentials', async () => {
    mockedLoggedModelCredentialsSupported.mockReturnValue(true);
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

  it.each([true, false])('signs direct logged-model reads when advertised (presigned-only: %s)', async (enforced) => {
    mockedPresignedOnly.mockReturnValue(enforced);
    mockedModelPresignedDownloadsSupported.mockReturnValue(true);
    const sign = jest
      .spyOn(MlflowService, 'getLoggedModelPresignedDownloadUrl')
      .mockResolvedValue({ url: 'https://storage.example/model', headers: { 'x-storage': 'required' } });
    const credentials = jest.spyOn(MlflowService, 'getCredentialsForLoggedModelArtifactRead');
    await expect(
      resolvePresignedArtifactDownload({
        runUuid: '',
        path: 'MLmodel',
        isLoggedModelsMode: true,
        loggedModelId: 'model-id',
        artifactRootUri: 's3://bucket/model/artifacts',
      }),
    ).resolves.toEqual({ url: 'https://storage.example/model', headers: { 'x-storage': 'required' } });
    expect(sign).toHaveBeenCalledWith({ loggedModelId: 'model-id', path: 'MLmodel' });
    expect(credentials).not.toHaveBeenCalled();
  });

  it('resolves a logged-model artifact without requiring its root URI', async () => {
    mockedLoggedModelCredentialsSupported.mockReturnValue(true);
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

  it('uses the existing logged-model artifact route when credential APIs are not advertised', async () => {
    const credentials = jest.spyOn(MlflowService, 'getCredentialsForLoggedModelArtifactRead');
    const sign = jest.spyOn(MlflowService, 'getLoggedModelPresignedDownloadUrl');
    const getArtifact = jest.fn(async (_url: string) => 'model contents');

    await expect(
      fetchArtifactPreferPresigned(
        {
          runUuid: '',
          path: 'MLmodel',
          artifactRootUri: 's3://bucket/logged-models/model-id/artifacts',
          isLoggedModelsMode: true,
          loggedModelId: 'model-id',
        },
        '/logged-models/model-id/artifacts/files?artifact_file_path=MLmodel',
        getArtifact,
      ),
    ).resolves.toBe('model contents');
    expect(credentials).not.toHaveBeenCalled();
    expect(sign).not.toHaveBeenCalled();
    expect(getArtifact).toHaveBeenCalledTimes(1);
    expect(getArtifact).toHaveBeenCalledWith('/logged-models/model-id/artifacts/files?artifact_file_path=MLmodel');
  });

  it('shares run-root discovery between concurrent reads and uploads through an explicit artifact service', async () => {
    const serviceBaseUrl = `${window.location.origin}/prefix/`;
    const artifactUri = `${serviceBaseUrl}api/2.0/mlflow-artifacts/artifacts/12/run/artifacts`;
    const getRun = jest.spyOn(MlflowService, 'getRun').mockResolvedValue({ run: { info: { artifactUri } } } as any);
    const sign = jest
      .spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl')
      .mockImplementation(async (path) => ({ url: `https://storage.example/${path}` }));
    const create = jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload').mockResolvedValue({
      upload_id: 'upload-id',
      credentials: [{ url: 'https://storage.example/upload', part_number: 1 }],
    });
    const complete = jest.spyOn(MlflowService, 'completeMlflowArtifactsMultipartUpload').mockResolvedValue({});
    const fetchPart = jest
      .spyOn(global, 'fetch')
      .mockResolvedValue(new Response(undefined, { headers: { ETag: 'part' } }));
    const getArtifact = jest.fn(async (url: string) => url);

    await Promise.all([
      fetchRunArtifactPreferPresigned('run', 'first.png', '/first', getArtifact),
      fetchRunArtifactPreferPresigned('run', 'second.png', '/second', getArtifact),
      uploadArtifactWithPresignedUrl('run', 'third.png', 'image'),
    ]);
    await fetchRunArtifactPreferPresigned('run', 'fourth.png', '/fourth', getArtifact);

    expect(getRun).toHaveBeenCalledTimes(1);
    expect(sign).toHaveBeenCalledTimes(3);
    expect(sign).toHaveBeenNthCalledWith(1, '12/run/artifacts/first.png', serviceBaseUrl);
    expect(sign).toHaveBeenNthCalledWith(2, '12/run/artifacts/second.png', serviceBaseUrl);
    expect(sign).toHaveBeenNthCalledWith(3, '12/run/artifacts/fourth.png', serviceBaseUrl);
    expect(getArtifact).toHaveBeenCalledTimes(3);
    expect(create).toHaveBeenCalledTimes(1);
    expect(complete).toHaveBeenCalledTimes(1);
    expect(fetchPart).toHaveBeenCalledTimes(1);
  });

  it('uses provided run roots without fetching run metadata', async () => {
    const getRun = jest.spyOn(MlflowService, 'getRun');
    const proxyUpload = jest.spyOn(MlflowService, 'uploadMlflowArtifact').mockResolvedValue({});
    const getArtifact = jest.fn(async (_url: string) => 'contents');
    const root = 'mlflow-artifacts:/12/run/artifacts';
    await expect(fetchRunArtifactPreferPresigned('run', 'file.txt', '/file', getArtifact, root)).resolves.toBe(
      'contents',
    );
    await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'contents', root)).resolves.toBe(true);
    expect(getRun).not.toHaveBeenCalled();
    expect(getArtifact).toHaveBeenCalledWith('ajax-api/2.0/mlflow-artifacts/artifacts/12/run/artifacts/file.txt');
    expect(proxyUpload).toHaveBeenCalledWith('12/run/artifacts/file.txt', 'contents', undefined);
  });

  it('separates cached run roots by run, workspace, authentication, and deployment', async () => {
    const getRun = jest
      .spyOn(MlflowService, 'getRun')
      .mockResolvedValue({ run: { info: { artifactUri: 'mlflow-artifacts:/12/run/artifacts' } } } as any);
    const getArtifact = jest.fn(async (_url: string) => 'contents');
    const read = (run = 'run') => fetchRunArtifactPreferPresigned(run, 'file.txt', '/file', getArtifact);
    await read();
    await read();
    expect(getRun).toHaveBeenCalledTimes(1);
    await read('another-run');
    setActiveWorkspace('other-workspace');
    await read();
    document.cookie = 'mlflow-request-header-Authorization=new-session; path=/';
    await read();
    const base = document.createElement('base');
    base.href = `${window.location.origin}/other-mlflow/`;
    document.head.append(base);
    try {
      await read();
      await read();
    } finally {
      base.remove();
    }
    expect(getRun).toHaveBeenCalledTimes(5);
    expect(getArtifact).toHaveBeenCalledTimes(7);
  });

  it('expires cached roots and retries rejected lookups', async () => {
    const getRun = jest
      .spyOn(MlflowService, 'getRun')
      .mockRejectedValueOnce(new ErrorWrapper('temporary metadata failure', 503))
      .mockResolvedValue({ run: { info: { artifactUri: 'mlflow-artifacts:/12/run/artifacts' } } } as any);
    const getArtifact = jest.fn(async (_url: string) => 'contents');
    const read = () => fetchRunArtifactPreferPresigned('run', 'file.txt', '/file', getArtifact);
    await expect(read()).rejects.toMatchObject({ status: 503 });
    await read();
    await read();
    expect(getRun).toHaveBeenCalledTimes(2);
    now += 60_001;
    await read();
    expect(getRun).toHaveBeenCalledTimes(3);
    expect(getArtifact).toHaveBeenCalledTimes(3);
  });

  it('bounds the number of cached run roots', async () => {
    const getRun = jest
      .spyOn(MlflowService, 'getRun')
      .mockResolvedValue({ run: { info: { artifactUri: 'mlflow-artifacts:/12/run/artifacts' } } } as any);
    const getArtifact = jest.fn(async (_url: string) => 'contents');
    for (let index = 0; index <= 100; index++) {
      await fetchRunArtifactPreferPresigned(`cached-run-${index}`, 'file.txt', '/file', getArtifact);
    }
    await fetchRunArtifactPreferPresigned('cached-run-0', 'file.txt', '/file', getArtifact);
    expect(getRun).toHaveBeenCalledTimes(102);
    expect(getArtifact).toHaveBeenCalledTimes(102);
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

  it('completes Azure Put Block uploads without requiring an ETag', async () => {
    mockedMultipartUploads.mockReturnValue(true);
    const getRun = jest.spyOn(MlflowService, 'getRun').mockResolvedValue({
      run: { info: { artifactUri: 'mlflow-artifacts:/12/run-id/artifacts' } },
    } as any);
    const url =
      'https://account.blob.core.windows.net/container/file?comp=block&blockid=encoded-block-id&sig=signature';
    const create = jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload').mockResolvedValue({
      upload_id: 'upload-id',
      credentials: [{ url, part_number: 1, headers: {} }],
    });
    const complete = jest.spyOn(MlflowService, 'completeMlflowArtifactsMultipartUpload').mockResolvedValue({});
    const abort = jest.spyOn(MlflowService, 'abortMlflowArtifactsMultipartUpload');
    const put = jest.spyOn(global, 'fetch').mockResolvedValue(new Response(undefined, { status: 201 }));

    await expect(uploadArtifactWithPresignedUrl('run-id', 'table.json', '{}')).resolves.toBe(true);
    expect(getRun).toHaveBeenCalledTimes(1);
    expect(create).toHaveBeenCalledTimes(1);
    expect(put).toHaveBeenCalledWith(url, { method: 'PUT', headers: {}, body: '{}' });
    expect(complete).toHaveBeenCalledWith('12/run-id/artifacts', {
      path: 'table.json',
      upload_id: 'upload-id',
      parts: [{ part_number: 1, etag: '', url }],
    });
    expect(abort).not.toHaveBeenCalled();
  });

  it.each([undefined, `${window.location.origin}/mlflow/`])(
    'aborts an upload when storage hides its ETag (artifact service: %s)',
    async (artifactServiceBaseUrl) => {
      mockedPresignedOnly.mockReturnValue(true);
      const artifactUri = artifactServiceBaseUrl
        ? `${artifactServiceBaseUrl}api/2.0/mlflow-artifacts/artifacts/12/run-id/artifacts`
        : 'mlflow-artifacts:/12/run-id/artifacts';
      jest.spyOn(MlflowService, 'getRun').mockResolvedValue({ run: { info: { artifactUri } } } as any);
      const createSpy = jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload').mockResolvedValue({
        upload_id: 'upload-id',
        credentials: [{ url: 'https://storage.example/part', part_number: 1, headers: {} }],
      });
      const completeSpy = jest.spyOn(MlflowService, 'completeMlflowArtifactsMultipartUpload').mockResolvedValue({});
      const abortSpy = jest.spyOn(MlflowService, 'abortMlflowArtifactsMultipartUpload').mockResolvedValue({});
      const fetchSpy = jest.spyOn(global, 'fetch').mockResolvedValue(new Response(undefined, { status: 200 }));

      await expect(uploadArtifactWithPresignedUrl('run-id', 'table.json', '{}')).rejects.toMatchObject({
        text: expect.stringContaining('ETag'),
        status: 500,
      });

      expect(createSpy).toHaveBeenCalledTimes(1);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect(completeSpy).not.toHaveBeenCalled();
      const abortArgs = ['12/run-id/artifacts', { path: 'table.json', upload_id: 'upload-id' }];
      expect(abortSpy).toHaveBeenCalledTimes(1);
      expect(abortSpy.mock.calls[0]).toEqual([
        ...abortArgs,
        ...(artifactServiceBaseUrl ? [artifactServiceBaseUrl] : []),
      ]);
    },
  );

  it('aborts an upload with missing credentials and preserves the failure if cleanup fails', async () => {
    mockedPresignedOnly.mockReturnValue(true);
    jest.spyOn(MlflowService, 'getRun').mockResolvedValue({
      run: { info: { artifactUri: 'mlflow-artifacts:/12/run-id/artifacts' } },
    } as any);
    const createSpy = jest.spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload').mockResolvedValue({
      upload_id: 'upload-id',
      credentials: [],
    });
    const completeSpy = jest.spyOn(MlflowService, 'completeMlflowArtifactsMultipartUpload').mockResolvedValue({});
    const abortSpy = jest
      .spyOn(MlflowService, 'abortMlflowArtifactsMultipartUpload')
      .mockRejectedValue(new Error('cleanup failed'));
    const fetchSpy = jest.spyOn(global, 'fetch');

    await expect(uploadArtifactWithPresignedUrl('run-id', 'table.json', '{}')).rejects.toMatchObject({
      text: expect.stringContaining('credential'),
      status: 501,
    });

    expect(createSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(completeSpy).not.toHaveBeenCalled();
    expect(abortSpy).toHaveBeenCalledWith('12/run-id/artifacts', { path: 'table.json', upload_id: 'upload-id' });
  });

  it.each(['unsupported signing', 'storage CORS', 'hidden ETag'] as const)(
    'keeps uploads in the original store after %s under a different service prefix',
    async (failure) => {
      const fixture = mockSeparateArtifactStores(failure);
      try {
        const getRun = jest.spyOn(MlflowService, 'getRun').mockResolvedValue({
          run: { info: { artifactUri: fixture.artifactRootUri } },
        } as any);
        const store = configureStore([thunk, promiseMiddleware()])({});
        await store.dispatch(uploadArtifactApi('run-id', 'table.json', { value: 'new contents' }));

        expect(getRun).toHaveBeenCalledTimes(1);
        expect(fixture.trackingStore.get('table.json')).toBe('contents from tracking store A');
        expect(fixture.artifactStore.get('table.json')).toBe('{"value":"new contents"}');
        expect(
          fixture.requests.filter(({ method, path }) => method === 'PUT' && path === fixture.proxyPath),
        ).toHaveLength(1);
        expect(fixture.requests.some(({ path }) => path.startsWith('/tracking/'))).toBe(false);
        expect(fixture.requests.some(({ path }) => path.includes('/mpu/complete/'))).toBe(false);
        expect(fixture.multipartParts.size).toBe(0);
        const abortIndex = fixture.requests.findIndex(({ path }) => path.includes('/mpu/abort/'));
        if (failure === 'unsupported signing') {
          expect(abortIndex).toBe(-1);
        } else {
          expect(abortIndex).toBeGreaterThan(-1);
          expect(abortIndex).toBeLessThan(fixture.requests.findIndex(({ path }) => path === fixture.proxyPath));
        }
        for (const request of fixture.requests.filter(({ path }) => path.startsWith('/artifacts/'))) {
          expect(request.headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
          expect(request.headers.get('Authorization')).toBe('tracking-secret');
        }
        for (const request of fixture.requests.filter(({ path }) => path === '/part')) {
          expect(request.headers.get('X-MLFLOW-WORKSPACE')).toBeNull();
          expect(request.headers.get('Authorization')).toBeNull();
        }
      } finally {
        fixture.cleanup();
      }
    },
  );

  it('keeps the committed destination correct when multipart abort cleanup fails', async () => {
    const fixture = mockSeparateArtifactStores('hidden ETag', true);
    try {
      await expect(
        uploadArtifactWithPresignedUrl('run-id', 'table.json', 'new contents', fixture.artifactRootUri),
      ).resolves.toBe(true);
      expect(fixture.artifactStore.get('table.json')).toBe('new contents');
      expect(fixture.trackingStore.get('table.json')).toBe('contents from tracking store A');
      // Failed cleanup leaves an uncommitted multipart part, never a committed
      // object in a different store. Storage lifecycle cleanup can remove the part.
      expect(fixture.multipartParts.get('upload-id')).toBe('new contents');
      const abortIndex = fixture.requests.findIndex(({ path }) => path.includes('/mpu/abort/'));
      expect(abortIndex).toBeGreaterThan(-1);
      expect(abortIndex).toBeLessThan(fixture.requests.findIndex(({ path }) => path === fixture.proxyPath));
      expect(fixture.requests.some(({ path }) => path.startsWith('/tracking/'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it.each(['unsupported signing', 'storage CORS'] as const)(
    'keeps downloads in the original store after %s under a different service prefix',
    async (failure) => {
      const fixture = mockSeparateArtifactStores(failure);
      try {
        await expect(
          fetchArtifactPreferPresigned(
            { runUuid: 'run-id', path: 'table.json', artifactRootUri: fixture.artifactRootUri },
            `${window.location.origin}/tracking/get-artifact?run_uuid=run-id&path=table.json`,
            fixture.readText,
          ),
        ).resolves.toBe('contents from artifact store B');
        expect(fixture.requests.some(({ path }) => path.startsWith('/tracking/'))).toBe(false);
        expect(fixture.requests.filter(({ path }) => path === fixture.proxyPath)).toHaveLength(1);
      } finally {
        fixture.cleanup();
      }
    },
  );

  it('discovers an omitted run root once before choosing a normal-mode fallback store', async () => {
    const fixture = mockSeparateArtifactStores('unsupported signing');
    try {
      const getRun = jest
        .spyOn(MlflowService, 'getRun')
        .mockResolvedValue({ run: { info: { artifactUri: fixture.artifactRootUri } } } as any);
      await expect(
        fetchArtifactPreferPresigned(
          { runUuid: 'run-id', path: 'table.json' },
          '/tracking/get-artifact',
          fixture.readText,
        ),
      ).resolves.toBe('contents from artifact store B');
      expect(getRun).toHaveBeenCalledTimes(1);
      expect(fixture.requests.some(({ path }) => path.startsWith('/tracking/'))).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  it.each([new TypeError('metadata transport failure'), new ErrorWrapper('missing metadata', 404)])(
    'does not guess a normal-mode download store after discovery failure: %s',
    async (error) => {
      const getRun = jest.spyOn(MlflowService, 'getRun').mockRejectedValue(error);
      const read = jest.fn<(url: string) => Promise<string>>();
      await expect(
        fetchArtifactPreferPresigned({ runUuid: 'run-id', path: 'table.json' }, '/tracking/get-artifact', read),
      ).rejects.toBe(error);
      expect(getRun).toHaveBeenCalledTimes(1);
      expect(read).not.toHaveBeenCalled();
    },
  );

  it.each([false, true])('rejects metadata without an artifact URI (logged model: %s)', async (loggedModel) => {
    const run = jest.spyOn(MlflowService, 'getRun').mockResolvedValue({ run: { info: {} } } as any);
    const model = jest.mocked(MlflowService.getLoggedModel).mockResolvedValue({ model: { info: {} } } as any);
    const read = jest.fn<(url: string) => Promise<string>>();
    await expect(
      fetchArtifactPreferPresigned(
        {
          runUuid: 'run-id',
          path: 'file',
          isLoggedModelsMode: loggedModel,
          loggedModelId: loggedModel ? 'm-1' : undefined,
        },
        '/tracking/get-artifact',
        read,
      ),
    ).rejects.toMatchObject({ status: 501, text: expect.stringContaining('artifact root') });
    expect(read).not.toHaveBeenCalled();
    expect(loggedModel ? model : run).toHaveBeenCalledTimes(1);
  });

  it('awaits local multipart abort before a normal-mode hidden-ETag proxy upload', async () => {
    mockedMultipartUploads.mockReturnValue(true);
    const create = jest
      .spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload')
      .mockResolvedValue({ upload_id: 'id', credentials: [{ url: 'https://storage.example/part', part_number: 1 }] });
    const part = jest.spyOn(global, 'fetch').mockResolvedValue(new Response(undefined, { status: 200 }));
    let finishAbort = () => {};
    let markAbortStarted = () => {};
    const abortStarted = new Promise<void>((resolve) => {
      markAbortStarted = resolve;
    });
    const abortFinished = new Promise<unknown>((resolve) => {
      finishAbort = () => resolve({});
    });
    const abort = jest.spyOn(MlflowService, 'abortMlflowArtifactsMultipartUpload').mockImplementation(() => {
      markAbortStarted();
      return abortFinished;
    });
    const proxy = jest.spyOn(MlflowService, 'uploadMlflowArtifact').mockResolvedValue({});
    const upload = uploadArtifactWithPresignedUrl(
      'run-id',
      'table.json',
      'raw bytes',
      'mlflow-artifacts:/custom/local/root',
    );
    await abortStarted;
    expect(proxy).not.toHaveBeenCalled();
    finishAbort();
    await expect(upload).resolves.toBe(true);
    expect(create).toHaveBeenCalledTimes(1);
    expect(part).toHaveBeenCalledTimes(1);
    expect(abort).toHaveBeenCalledTimes(1);
    expect(proxy).toHaveBeenCalledWith('custom/local/root/table.json', 'raw bytes', undefined);
  });

  it('reuses one discovered logged-model root when signing falls back', async () => {
    const root = `${window.location.origin}/artifacts/api/2.0/mlflow-artifacts/artifacts/custom/model/root`;
    const model = jest
      .mocked(MlflowService.getLoggedModel)
      .mockResolvedValue({ model: { info: { artifact_uri: root } } });
    const sign = jest
      .spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl')
      .mockRejectedValue(new ErrorWrapper('unsupported', 501));
    const read = jest.fn<(url: string) => Promise<string>>().mockResolvedValue('model contents');
    await expect(
      fetchArtifactPreferPresigned(
        { runUuid: '', path: 'MLmodel', isLoggedModelsMode: true, loggedModelId: 'm-1' },
        '/tracking/model-files',
        read,
      ),
    ).resolves.toBe('model contents');
    expect(model).toHaveBeenCalledTimes(1);
    expect(sign).toHaveBeenCalledWith('custom/model/root/MLmodel', `${window.location.origin}/artifacts/`);
    expect(read).toHaveBeenCalledWith(
      `${window.location.origin}/artifacts/api/2.0/mlflow-artifacts/artifacts/custom/model/root/MLmodel`,
    );
  });

  it.each([401, 403, 500])('does not convert a genuine upload %s into proxy fallback', async (status) => {
    mockedMultipartUploads.mockReturnValue(true);
    const root = 'mlflow-artifacts:/custom/root/run-id/artifacts';
    const create = jest
      .spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload')
      .mockResolvedValue({ upload_id: 'id', credentials: [{ url: 'https://storage.example/part', part_number: 1 }] });
    const put = jest.spyOn(global, 'fetch').mockResolvedValue(new Response('storage error', { status }));
    const abort = jest.spyOn(MlflowService, 'abortMlflowArtifactsMultipartUpload').mockResolvedValue({});
    const proxy = jest.spyOn(MlflowService, 'uploadMlflowArtifact');
    await expect(uploadArtifactWithPresignedUrl('run-id', 'table.json', '{}', root)).rejects.toMatchObject({ status });
    expect(create).toHaveBeenCalledTimes(1);
    expect(put).toHaveBeenCalledTimes(1);
    expect(abort).toHaveBeenCalledTimes(1);
    expect(proxy).not.toHaveBeenCalled();
  });

  it('does not retry a failed compatibility upload or switch back to tracking', async () => {
    const error = new ErrorWrapper('missing byte route', 404);
    const proxy = jest.spyOn(MlflowService, 'uploadMlflowArtifact').mockRejectedValue(error);
    await expect(
      uploadArtifactWithPresignedUrl('run-id', 'table.json', '{}', 'mlflow-artifacts:/custom/run/root'),
    ).rejects.toBe(error);
    expect(proxy).toHaveBeenCalledTimes(1);
  });

  it('preserves literal percent signs alongside valid UTF-8 escapes and separators', async () => {
    const root = `${window.location.origin}/artifacts/api/2.0/mlflow-artifacts/artifacts/100%complete/a%20b/%E2%82%AC/encoded%2Fchild`;
    const expected = `${window.location.origin}/artifacts/api/2.0/mlflow-artifacts/artifacts/100%25complete/a%20b/%E2%82%AC/encoded/child/file.txt`;
    const sign = jest
      .spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl')
      .mockRejectedValue(new ErrorWrapper('unsupported', 501));
    const create = jest
      .spyOn(MlflowService, 'createMlflowArtifactsMultipartUpload')
      .mockRejectedValue(new ErrorWrapper('unsupported', 501));
    const fetch = jest.spyOn(global, 'fetch').mockResolvedValue(new Response('{}'));
    const read = jest.fn(async (url: string) => url);
    await expect(
      fetchArtifactPreferPresigned(
        { runUuid: 'run', path: 'file.txt', artifactRootUri: root },
        '/tracking/get-artifact',
        read,
      ),
    ).resolves.toBe(expected);
    await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'raw contents', root)).resolves.toBe(true);
    expect(sign).toHaveBeenCalledWith(
      '100%complete/a b/€/encoded/child/file.txt',
      `${window.location.origin}/artifacts/`,
    );
    expect(create).toHaveBeenCalledWith(
      '100%complete/a b/€/encoded/child',
      { path: 'file.txt', num_parts: 1 },
      `${window.location.origin}/artifacts/`,
    );
    expect(fetch).toHaveBeenCalledWith(expected, expect.objectContaining({ method: 'PUT', body: 'raw contents' }));
    expect(read).toHaveBeenCalledTimes(1);
  });

  it.each([
    { root: `${window.location.origin}/artifacts/api/2.0/mlflow-artifacts/artifacts/%E9/root`, status: 400 },
    { root: 'https://external.example/api/2.0/mlflow-artifacts/artifacts/%E9/root', status: 400 },
    { root: 'https://external.example/api/2.0/mlflow-artifacts/artifacts/100%complete/root', status: 403 },
    { root: 'mlflow-artifacts://%bad/root', status: 400 },
  ])(
    'fails closed for an unsafe proxied root without direct signing or tracking fallback: $root',
    async ({ root, status }) => {
      const read = jest.fn<(url: string) => Promise<string>>();
      const directSign = jest.spyOn(MlflowService, 'createPresignedDownloadUrl');
      const directUpload = jest.spyOn(MlflowService, 'createPresignedUploadUrl');
      const fetch = jest.spyOn(global, 'fetch');
      await expect(
        fetchArtifactPreferPresigned(
          { runUuid: 'run', path: 'file.txt', artifactRootUri: root },
          '/tracking/get-artifact',
          read,
        ),
      ).rejects.toMatchObject({ status });
      await expect(uploadArtifactWithPresignedUrl('run', 'file.txt', 'data', root)).rejects.toMatchObject({ status });
      expect(read).not.toHaveBeenCalled();
      expect(directSign).not.toHaveBeenCalled();
      expect(directUpload).not.toHaveBeenCalled();
      expect(fetch).not.toHaveBeenCalled();
    },
  );
});

/** A reverse proxy serves tracking store A and artifact store B under one origin. */
const mockSeparateArtifactStores = (
  failure: 'unsupported signing' | 'storage CORS' | 'hidden ETag',
  abortFails = false,
) => {
  const base = document.createElement('base');
  base.href = `${window.location.origin}/tracking/`;
  document.head.append(base);
  const previousAbsoluteUrls = process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
  process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'false';
  setActiveWorkspace('team-a');
  document.cookie = 'mlflow-request-header-Authorization=tracking-secret; path=/';
  mockedMultipartUploads.mockReturnValue(true);
  const artifactRootUri = `${window.location.origin}/artifacts/api/2.0/mlflow-artifacts/artifacts/custom/root/run-id/artifacts`;
  const proxyPath = '/artifacts/api/2.0/mlflow-artifacts/artifacts/custom/root/run-id/artifacts/table.json';
  const trackingStore = new Map([['table.json', 'contents from tracking store A']]);
  const artifactStore = new Map([['table.json', 'contents from artifact store B']]);
  const multipartParts = new Map<string, string>();
  const requests: { method: string; path: string; headers: Headers }[] = [];
  jest.spyOn(global, 'fetch').mockImplementation(async (input, options) => {
    const request =
      input instanceof Request ? input : new Request(new URL(String(input), document.baseURI).toString(), options);
    const url = new URL(request.url, document.baseURI);
    requests.push({ method: request.method, path: url.pathname, headers: request.headers });
    if (url.origin === 'https://storage.example') {
      if (request.method === 'GET') throw new TypeError('Storage CORS denied');
      multipartParts.set('upload-id', await request.text());
      if (failure === 'storage CORS') throw new TypeError('Storage CORS denied');
      return new Response(undefined, { status: 200 });
    }
    if (url.pathname.includes('/presigned/') || url.pathname.includes('/mpu/create/')) {
      if (failure === 'unsupported signing') return new Response('{"message":"unsupported"}', { status: 501 });
      return new Response(
        JSON.stringify(
          request.method === 'GET'
            ? { url: 'https://storage.example/part' }
            : { upload_id: 'upload-id', credentials: [{ url: 'https://storage.example/part', part_number: 1 }] },
        ),
      );
    }
    if (url.pathname.includes('/mpu/abort/')) {
      if (abortFails) return new Response('{"message":"abort unavailable"}', { status: 500 });
      multipartParts.clear();
      return new Response('{}');
    }
    if (url.pathname === proxyPath) {
      if (request.method === 'PUT') artifactStore.set('table.json', await request.text());
      return new Response(request.method === 'PUT' ? '{}' : artifactStore.get('table.json'));
    }
    if (url.pathname.startsWith('/tracking/')) {
      if (request.method === 'POST') trackingStore.set('table.json', await request.text());
      return new Response(request.method === 'POST' ? '{}' : trackingStore.get('table.json'));
    }
    throw new Error(`Unexpected request: ${request.method} ${url}`);
  });
  const readText = async (url: string, options?: { headers?: HeadersInit }) => {
    const response = await global.fetch(
      new Request(new URL(url, document.baseURI).toString(), {
        headers: options?.headers ?? getDefaultHeaders(document.cookie),
      }),
    );
    if (!response.ok) throw new ErrorWrapper(await response.text(), response.status);
    return response.text();
  };
  return {
    artifactRootUri,
    proxyPath,
    trackingStore,
    artifactStore,
    multipartParts,
    requests,
    readText,
    cleanup: () => {
      base.remove();
      if (previousAbsoluteUrls === undefined) delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
      else process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = previousAbsoluteUrls;
    },
  };
};
