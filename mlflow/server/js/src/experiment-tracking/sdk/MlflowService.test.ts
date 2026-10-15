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

  it.each(operations)('does not forward tracking credentials for external %s requests', async (_name, operation) => {
    await operation('https://artifacts.example/prefix/');
    const [url, options] = jest.mocked(global.fetch).mock.calls[0];
    expect(url).toEqual(expect.stringContaining('https://artifacts.example/prefix/api/'));
    const headers = new Headers(options?.headers);
    expect(headers.get('Authorization')).toBeNull();
    expect(headers.get('X-MLFLOW-WORKSPACE')).toBeNull();
    expect(headers.get('Content-Type')).toContain('application/json');
    expect(options?.credentials).toBe('omit');
  });

  it.each(operations)('retains tracking credentials for same-origin %s requests', async (_name, operation) => {
    await operation(`${window.location.origin}/prefix/`);
    const [, options] = jest.mocked(global.fetch).mock.calls[0];
    const headers = new Headers(options?.headers);
    expect(headers.get('Authorization')).toBe('secret');
    expect(headers.get('X-MLFLOW-WORKSPACE')).toBe('team-a');
  });
});
