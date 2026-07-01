import fs from 'fs';
import os from 'os';
import path from 'path';
import { createAuthProvider, createKubernetesAuth } from '../../src/auth';

const ENV_KEYS = [
  'MLFLOW_TRACKING_TOKEN',
  'MLFLOW_TRACKING_USERNAME',
  'MLFLOW_TRACKING_PASSWORD',
  'MLFLOW_WORKSPACE',
  'MLFLOW_TRACKING_AUTH',
] as const;

describe('createAuthProvider', () => {
  describe('OSS auth (createOssAuth)', () => {
    const savedEnv: Record<string, string | undefined> = {};

    beforeEach(() => {
      for (const key of ENV_KEYS) {
        savedEnv[key] = process.env[key];
        delete process.env[key];
      }
    });

    afterEach(() => {
      for (const key of ENV_KEYS) {
        if (savedEnv[key] === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = savedEnv[key];
        }
      }
    });

    it('should include Bearer token header when MLFLOW_TRACKING_TOKEN is set', async () => {
      process.env.MLFLOW_TRACKING_TOKEN = 'test-token';

      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer test-token');
      expect(headers['Content-Type']).toBe('application/json');
    });

    it('should include Basic auth header when username and password are set', async () => {
      process.env.MLFLOW_TRACKING_USERNAME = 'user';
      process.env.MLFLOW_TRACKING_PASSWORD = 'pass';

      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });
      const headers = await provider.getHeadersProvider()();

      const expected = `Basic ${Buffer.from('user:pass').toString('base64')}`;
      expect(headers['Authorization']).toBe(expected);
    });

    it('should not include Authorization header when no credentials are set', async () => {
      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBeUndefined();
      expect(headers['Content-Type']).toBe('application/json');
    });

    it('should include X-MLFLOW-WORKSPACE header when MLFLOW_WORKSPACE is set', async () => {
      process.env.MLFLOW_WORKSPACE = 'my-namespace';

      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('my-namespace');
    });

    it('should not include X-MLFLOW-WORKSPACE header when MLFLOW_WORKSPACE is not set', async () => {
      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBeUndefined();
    });

    it('should include both Authorization and X-MLFLOW-WORKSPACE when both are set', async () => {
      process.env.MLFLOW_TRACKING_TOKEN = 'test-token';
      process.env.MLFLOW_WORKSPACE = 'my-namespace';

      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer test-token');
      expect(headers['X-MLFLOW-WORKSPACE']).toBe('my-namespace');
      expect(headers['Content-Type']).toBe('application/json');
    });

    it('should use workspace from options when env var is not set', async () => {
      const provider = createAuthProvider({
        trackingUri: 'http://localhost:5000',
        workspace: 'config-namespace',
      });
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('config-namespace');
    });

    it('should prefer MLFLOW_WORKSPACE env var over options', async () => {
      process.env.MLFLOW_WORKSPACE = 'env-namespace';

      const provider = createAuthProvider({
        trackingUri: 'http://localhost:5000',
        workspace: 'config-namespace',
      });
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('env-namespace');
    });

    it('should return the tracking URI as host', () => {
      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });

      expect(provider.getHost()).toBe('http://localhost:5000');
    });

    it('should return undefined for getDatabricksToken', () => {
      const provider = createAuthProvider({ trackingUri: 'http://localhost:5000' });

      expect(provider.getDatabricksToken()).toBeUndefined();
    });
  });

  // Tests prefixed "kubernetes:" test MLFLOW_TRACKING_AUTH=kubernetes path
  // Tests prefixed "kubernetes-namespaced:" test MLFLOW_TRACKING_AUTH=kubernetes-namespaced path
  describe('Kubernetes auth (MLFLOW_TRACKING_AUTH)', () => {
    const savedEnv: Record<string, string | undefined> = {};
    let tmpDir: string;

    beforeEach(() => {
      for (const key of ENV_KEYS) {
        savedEnv[key] = process.env[key];
        delete process.env[key];
      }
      tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mlflow-auth-test-'));
    });

    afterEach(() => {
      for (const key of ENV_KEYS) {
        if (savedEnv[key] === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = savedEnv[key];
        }
      }
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    // --- Token resolution ---

    it('kubernetes: reads bearer token from SA file', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token-abc123');

      const provider = createKubernetesAuth('http://mlflow:5000', false, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token-abc123');
      expect(headers['Content-Type']).toBe('application/json');
    });

    it('kubernetes: throws when token file is missing and no kubeconfig', async () => {
      const tokenPath = path.join(tmpDir, 'nonexistent-token');

      const provider = createKubernetesAuth('http://mlflow:5000', false, tokenPath);

      await expect(provider.getHeadersProvider()()).rejects.toThrow(
        'Could not determine Kubernetes credentials',
      );
    });

    it('kubernetes: caches token reads within TTL (60s)', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'old-token');

      const provider = createKubernetesAuth('http://mlflow:5000', false, tokenPath);

      const headers1 = await provider.getHeadersProvider()();
      expect(headers1['Authorization']).toBe('Bearer old-token');

      // Kubelet rotates token on disk — cached value still returned within TTL
      fs.writeFileSync(tokenPath, 'rotated-token');
      const headers2 = await provider.getHeadersProvider()();
      expect(headers2['Authorization']).toBe('Bearer old-token');
    });

    it('kubernetes: strips whitespace and newlines from token file', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, '  sa-token-trimmed  \n');

      const provider = createKubernetesAuth('http://mlflow:5000', false, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token-trimmed');
    });

    it('kubernetes: file-based token takes priority over MLFLOW_TRACKING_TOKEN', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-file-token');
      process.env.MLFLOW_TRACKING_TOKEN = 'static-env-token';

      const provider = createKubernetesAuth('http://mlflow:5000', false, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-file-token');
    });

    // --- Workspace resolution ---

    it('kubernetes: no workspace header when MLFLOW_WORKSPACE is not set', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token');

      const provider = createKubernetesAuth('http://mlflow:5000', false, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token');
      expect(headers['X-MLFLOW-WORKSPACE']).toBeUndefined();
    });

    it('kubernetes: honors programmatic workspace option', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token');

      // namespacePath is undefined — irrelevant because enableWorkspaces is false
      const provider = createKubernetesAuth(
        'http://mlflow:5000', false, tokenPath, undefined, 'options-namespace',
      );
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('options-namespace');
    });

    it('kubernetes: honors MLFLOW_WORKSPACE even without kubernetes-namespaced', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token');
      process.env.MLFLOW_WORKSPACE = 'explicit-namespace';

      const provider = createKubernetesAuth('http://mlflow:5000', false, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token');
      expect(headers['X-MLFLOW-WORKSPACE']).toBe('explicit-namespace');
    });

    it('kubernetes-namespaced: reads namespace from SA file automatically', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      const namespacePath = path.join(tmpDir, 'namespace');
      fs.writeFileSync(tokenPath, 'sa-token-abc123');
      fs.writeFileSync(namespacePath, 'my-namespace');

      const provider = createKubernetesAuth('http://mlflow:5000', true, tokenPath, namespacePath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token-abc123');
      expect(headers['X-MLFLOW-WORKSPACE']).toBe('my-namespace');
    });

    it('kubernetes-namespaced: throws when namespace file is missing and no kubeconfig', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      const namespacePath = path.join(tmpDir, 'nonexistent-namespace');
      fs.writeFileSync(tokenPath, 'sa-token-abc123');

      const provider = createKubernetesAuth('http://mlflow:5000', true, tokenPath, namespacePath);

      await expect(provider.getHeadersProvider()()).rejects.toThrow(
        'Could not determine Kubernetes namespace',
      );
    });

    it('kubernetes-namespaced: strips whitespace from namespace file', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      const namespacePath = path.join(tmpDir, 'namespace');
      fs.writeFileSync(tokenPath, 'sa-token');
      fs.writeFileSync(namespacePath, '  my-ns  \n');

      const provider = createKubernetesAuth('http://mlflow:5000', true, tokenPath, namespacePath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('my-ns');
    });

    it('kubernetes-namespaced: MLFLOW_WORKSPACE overrides auto-discovered namespace', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      const namespacePath = path.join(tmpDir, 'namespace');
      fs.writeFileSync(tokenPath, 'sa-token');
      fs.writeFileSync(namespacePath, 'auto-namespace');
      process.env.MLFLOW_WORKSPACE = 'explicit-namespace';

      const provider = createKubernetesAuth('http://mlflow:5000', true, tokenPath, namespacePath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('explicit-namespace');
    });

    // --- Provider metadata ---

    it('returns the tracking URI as host and undefined for databricksToken', () => {
      const provider = createKubernetesAuth('http://mlflow:5000', false);

      expect(provider.getHost()).toBe('http://mlflow:5000');
      expect(provider.getDatabricksToken()).toBeUndefined();
    });
  });
});
