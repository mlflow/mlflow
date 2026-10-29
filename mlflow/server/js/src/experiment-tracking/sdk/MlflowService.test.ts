import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { MlflowService } from './MlflowService';

const operations = [
  ['download', (base: string) => MlflowService.getMlflowArtifactsPresignedDownloadUrl('run/file.txt', base)],
  [
    'create',
    (base: string) =>
      MlflowService.createMlflowArtifactsMultipartUpload('run', { path: 'file.txt', num_parts: 1 }, base),
  ],
  [
    'complete',
    (base: string) =>
      MlflowService.completeMlflowArtifactsMultipartUpload('run', { path: 'file.txt', parts: [] }, base),
  ],
  ['abort', (base: string) => MlflowService.abortMlflowArtifactsMultipartUpload('run', { path: 'file.txt' }, base)],
  ['proxy upload', (base: string) => MlflowService.uploadMlflowArtifact('run/file.txt', 'contents', base)],
] as const;

describe('artifact-service credential scope', () => {
  beforeEach(() => {
    document.cookie = 'mlflow-request-header-Authorization=secret; path=/';
    setActiveWorkspace('team-a');
    jest.spyOn(global, 'fetch').mockImplementation(async () => new Response('{}', { status: 200 }));
  });

  afterEach(() => {
    document.cookie = 'mlflow-request-header-Authorization=; Max-Age=0; path=/';
    setActiveWorkspace(null);
    jest.restoreAllMocks();
  });

  it.each(operations)(
    'rejects external %s requests before sending any credentials or network request',
    async (_name, operation) => {
      await expect(operation('https://artifacts.example/prefix/')).rejects.toMatchObject({
        status: 403,
        textJson: {
          external_artifact_service: true,
          message: expect.stringContaining('Configure a reverse proxy'),
        },
      });
      expect(global.fetch).not.toHaveBeenCalled();
    },
  );

  it.each(['port', 'scheme'])('rejects an artifact service with a different %s', async (difference) => {
    const base = new URL('/prefix/', window.location.href);
    if (difference === 'port') base.port = '9000';
    else base.protocol = base.protocol === 'http:' ? 'https:' : 'http:';
    await expect(
      MlflowService.getMlflowArtifactsPresignedDownloadUrl('run/file.txt', base.toString()),
    ).rejects.toMatchObject({
      status: 403,
      textJson: { external_artifact_service: true },
    });
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it.each(operations)('retains tracking credentials for same-origin %s requests', async (_name, operation) => {
    await operation(`${window.location.origin}/prefix/`);
    const [, options] = jest.mocked(global.fetch).mock.calls[0];
    const headers = new Headers(options?.headers);
    expect(headers.get('Authorization')).toBe('secret');
    expect(headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
    expect(jest.mocked(global.fetch).mock.calls[0][0]).toEqual(
      expect.stringContaining(`${window.location.origin}/prefix/api/`),
    );
  });

  it('retains tracking credentials on relative artifact proxy routes', async () => {
    await MlflowService.getMlflowArtifactsPresignedDownloadUrl('run/file.txt');
    const [url, options] = jest.mocked(global.fetch).mock.calls[0];
    expect(url).toBe('ajax-api/2.0/mlflow-artifacts/presigned/run/file.txt');
    const headers = new Headers(options?.headers);
    expect(headers.get('Authorization')).toBe('secret');
    expect(headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
  });

  it('rejects external byte URLs even when signing is disabled', () => {
    expect(() => MlflowService.getMlflowArtifactsProxyUrl('root/file.txt', 'https://artifacts.example/')).toThrow();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('preserves the service prefix and escapes every byte-route path segment', async () => {
    const base = `${window.location.origin}/artifacts/`;
    const path = 'custom root/run/a?#.json';
    const url = `${base}api/2.0/mlflow-artifacts/artifacts/custom%20root/run/a%3F%23.json`;
    expect(MlflowService.getMlflowArtifactsProxyUrl(path, base)).toBe(url);
    await MlflowService.uploadMlflowArtifact(path, 'raw contents', base);
    expect(global.fetch).toHaveBeenCalledWith(url, expect.objectContaining({ method: 'PUT', body: 'raw contents' }));
  });
});

describe('artifact proxy URLs for subpath UIs', () => {
  const originalUseAbsoluteAjaxUrls = process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
  let base: HTMLBaseElement;

  beforeEach(() => {
    base = document.createElement('base');
    base.href = new URL('/tracking/experiments/1', window.location.href).toString();
    document.head.append(base);
  });

  afterEach(() => {
    base.remove();
    if (originalUseAbsoluteAjaxUrls === undefined) delete process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'];
    else process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = originalUseAbsoluteAjaxUrls;
  });

  it.each([false, true])('resolves authority-free byte URLs with absolute AJAX set to %s', (absolute) => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = String(absolute);
    const route = 'ajax-api/2.0/mlflow-artifacts/artifacts/root/a%20b.txt';
    const url = MlflowService.getMlflowArtifactsProxyUrl('root/a b.txt');
    expect(url).toBe(absolute ? `/${route}` : route);
    expect(new URL(url, document.baseURI).pathname).toBe(absolute ? `/${route}` : `/tracking/experiments/${route}`);
  });

  it.each([false, true])('preserves explicit service prefixes with absolute AJAX set to %s', (absolute) => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = String(absolute);
    const base = `${window.location.origin}/artifact-service/`;
    expect(MlflowService.getMlflowArtifactsProxyUrl('root/a b.txt', base)).toBe(
      `${base}api/2.0/mlflow-artifacts/artifacts/root/a%20b.txt`,
    );
  });
});
