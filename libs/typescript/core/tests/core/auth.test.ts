import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  createAuthProvider,
  createKubernetesAuth,
  _setKubeconfigLoader,
  _setKubeconfigCacheKeyFn,
} from '../../src/auth';

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
    const ENABLE_WORKSPACES = true;
    const DISABLE_WORKSPACES = false;

    const savedEnv: Record<string, string | undefined> = {};
    let tmpDir: string;
    let nonExistentTokenPath: string;
    let nonExistentNamespacePath: string;

    beforeEach(() => {
      for (const key of ENV_KEYS) {
        savedEnv[key] = process.env[key];
        delete process.env[key];
      }
      tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mlflow-auth-test-'));
      nonExistentTokenPath = path.join(tmpDir, 'no-sa-token');
      nonExistentNamespacePath = path.join(tmpDir, 'no-sa-ns');
    });

    afterEach(() => {
      _setKubeconfigLoader(null);
      _setKubeconfigCacheKeyFn(null);
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

      const provider = createKubernetesAuth('http://mlflow:5000', DISABLE_WORKSPACES, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token-abc123');
      expect(headers['Content-Type']).toBe('application/json');
    });

    it('kubernetes: propagates kubeconfig error when SA token file is missing', async () => {
      _setKubeconfigLoader(() => {
        throw new Error('kubeconfig not found');
      });

      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        DISABLE_WORKSPACES,
        nonExistentTokenPath,
        nonExistentNamespacePath,
      );

      await expect(provider.getHeadersProvider()()).rejects.toThrow('kubeconfig not found');
    });

    it('kubernetes: caches token reads within TTL (60s)', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'old-token');

      const provider = createKubernetesAuth('http://mlflow:5000', DISABLE_WORKSPACES, tokenPath);

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

      const provider = createKubernetesAuth('http://mlflow:5000', DISABLE_WORKSPACES, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token-trimmed');
    });

    it('kubernetes: file-based token takes priority over MLFLOW_TRACKING_TOKEN', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-file-token');
      process.env.MLFLOW_TRACKING_TOKEN = 'static-env-token';

      const provider = createKubernetesAuth('http://mlflow:5000', DISABLE_WORKSPACES, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-file-token');
    });

    // --- Workspace resolution ---

    it('kubernetes: no workspace header when MLFLOW_WORKSPACE is not set', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token');

      const provider = createKubernetesAuth('http://mlflow:5000', DISABLE_WORKSPACES, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token');
      expect(headers['X-MLFLOW-WORKSPACE']).toBeUndefined();
    });

    it('kubernetes: honors programmatic workspace option', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token');

      // namespacePath is undefined — irrelevant because enableWorkspaces is false
      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        DISABLE_WORKSPACES,
        tokenPath,
        undefined,
        'options-namespace',
      );
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('options-namespace');
    });

    it('kubernetes: honors MLFLOW_WORKSPACE even without kubernetes-namespaced', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token');
      process.env.MLFLOW_WORKSPACE = 'explicit-namespace';

      const provider = createKubernetesAuth('http://mlflow:5000', DISABLE_WORKSPACES, tokenPath);
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token');
      expect(headers['X-MLFLOW-WORKSPACE']).toBe('explicit-namespace');
    });

    it('kubernetes-namespaced: reads namespace from SA file automatically', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      const namespacePath = path.join(tmpDir, 'namespace');
      fs.writeFileSync(tokenPath, 'sa-token-abc123');
      fs.writeFileSync(namespacePath, 'my-namespace');

      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        ENABLE_WORKSPACES,
        tokenPath,
        namespacePath,
      );
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer sa-token-abc123');
      expect(headers['X-MLFLOW-WORKSPACE']).toBe('my-namespace');
    });

    it('kubernetes-namespaced: propagates kubeconfig error when SA namespace file is missing', async () => {
      _setKubeconfigLoader(() => {
        throw new Error('kubeconfig not found');
      });

      const tokenPath = path.join(tmpDir, 'token');
      fs.writeFileSync(tokenPath, 'sa-token-abc123');

      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        ENABLE_WORKSPACES,
        tokenPath,
        nonExistentNamespacePath,
      );

      await expect(provider.getHeadersProvider()()).rejects.toThrow('kubeconfig not found');
    });

    it('kubernetes-namespaced: strips whitespace from namespace file', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      const namespacePath = path.join(tmpDir, 'namespace');
      fs.writeFileSync(tokenPath, 'sa-token');
      fs.writeFileSync(namespacePath, '  my-ns  \n');

      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        ENABLE_WORKSPACES,
        tokenPath,
        namespacePath,
      );
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('my-ns');
    });

    it('kubernetes-namespaced: MLFLOW_WORKSPACE overrides auto-discovered namespace', async () => {
      const tokenPath = path.join(tmpDir, 'token');
      const namespacePath = path.join(tmpDir, 'namespace');
      fs.writeFileSync(tokenPath, 'sa-token');
      fs.writeFileSync(namespacePath, 'auto-namespace');
      process.env.MLFLOW_WORKSPACE = 'explicit-namespace';

      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        ENABLE_WORKSPACES,
        tokenPath,
        namespacePath,
      );
      const headers = await provider.getHeadersProvider()();

      expect(headers['X-MLFLOW-WORKSPACE']).toBe('explicit-namespace');
    });

    // --- Kubeconfig fallback ---
    // Note: @kubernetes/client-node uses ESM which Jest can't transform directly.
    // These tests mock the module to verify the fallback and TTL logic without
    // requiring the real package in the test environment.

    it('kubernetes-namespaced: falls back to kubeconfig when SA files are missing', async () => {
      const mockLoader = jest.fn().mockResolvedValue({
        token: 'kubeconfig-token',
        namespace: 'kubeconfig-ns',
      });
      _setKubeconfigLoader(mockLoader);

      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        ENABLE_WORKSPACES,
        nonExistentTokenPath,
        nonExistentNamespacePath,
      );
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer kubeconfig-token');
      expect(headers['X-MLFLOW-WORKSPACE']).toBe('kubeconfig-ns');
      expect(mockLoader).toHaveBeenCalledTimes(1);
    });

    it('kubernetes-namespaced: picks up kubeconfig context changes immediately via cache key', async () => {
      const mockLoader = jest
        .fn()
        .mockResolvedValueOnce({ token: 'token-a', namespace: 'namespace-a' })
        .mockResolvedValueOnce({ token: 'token-b', namespace: 'namespace-b' });
      _setKubeconfigLoader(mockLoader);

      // eslint-disable-next-line require-await, @typescript-eslint/require-await
      _setKubeconfigCacheKeyFn(async () => 'DEFAULT:context-a');

      const provider = createKubernetesAuth(
        'http://mlflow:5000',
        ENABLE_WORKSPACES,
        nonExistentTokenPath,
        nonExistentNamespacePath,
      );

      // First call reads context A
      const headers1 = await provider.getHeadersProvider()();
      expect(headers1['Authorization']).toBe('Bearer token-a');
      expect(headers1['X-MLFLOW-WORKSPACE']).toBe('namespace-a');

      // Same cache key — returns cached context A
      const headers2 = await provider.getHeadersProvider()();
      expect(headers2['Authorization']).toBe('Bearer token-a');
      expect(mockLoader).toHaveBeenCalledTimes(1);

      // Simulate context switch (oc project / kubectl config use-context)
      // eslint-disable-next-line require-await, @typescript-eslint/require-await
      _setKubeconfigCacheKeyFn(async () => 'DEFAULT:context-b');

      // Invalidates immediately without waiting for TTL
      const headers3 = await provider.getHeadersProvider()();
      expect(headers3['Authorization']).toBe('Bearer token-b');
      expect(headers3['X-MLFLOW-WORKSPACE']).toBe('namespace-b');
      expect(mockLoader).toHaveBeenCalledTimes(2);
    });

    it('kubernetes-namespaced: refreshes kubeconfig after TTL within same context', async () => {
      const mockLoader = jest
        .fn()
        .mockResolvedValueOnce({ token: 'token-v1', namespace: 'ns' })
        .mockResolvedValueOnce({ token: 'token-v2', namespace: 'ns' });
      _setKubeconfigLoader(mockLoader);
      // eslint-disable-next-line require-await, @typescript-eslint/require-await
      _setKubeconfigCacheKeyFn(async () => 'DEFAULT:same-context');

      jest.useFakeTimers();
      try {
        const provider = createKubernetesAuth(
          'http://mlflow:5000',
          ENABLE_WORKSPACES,
          nonExistentTokenPath,
          nonExistentNamespacePath,
        );

        const headers1 = await provider.getHeadersProvider()();
        expect(headers1['Authorization']).toBe('Bearer token-v1');

        // 10s later, still under 60s TTL — cached
        jest.advanceTimersByTime(10_000);
        const headers2 = await provider.getHeadersProvider()();
        expect(headers2['Authorization']).toBe('Bearer token-v1');
        expect(mockLoader).toHaveBeenCalledTimes(1);

        // Advance past TTL — token rotation picked up
        jest.advanceTimersByTime(61_000);
        const headers3 = await provider.getHeadersProvider()();
        expect(headers3['Authorization']).toBe('Bearer token-v2');
        expect(mockLoader).toHaveBeenCalledTimes(2);
      } finally {
        jest.useRealTimers();
      }
    });

    // --- Provider metadata ---

    it('returns the tracking URI as host and undefined for databricksToken', () => {
      const provider = createKubernetesAuth('http://mlflow:5000', DISABLE_WORKSPACES);

      expect(provider.getHost()).toBe('http://mlflow:5000');
      expect(provider.getDatabricksToken()).toBeUndefined();
    });
  });

  // Tests that verify createAuthProvider routes correctly when
  // MLFLOW_TRACKING_AUTH=kubernetes* is combined with explicit credentials.
  // Tokens pass through; username/password are ignored (basic auth doesn't
  // overlap with kubernetes auth).
  describe('Auth precedence (createAuthProvider)', () => {
    const savedEnv: Record<string, string | undefined> = {};

    beforeEach(() => {
      for (const key of ENV_KEYS) {
        savedEnv[key] = process.env[key];
        delete process.env[key];
      }
    });

    afterEach(() => {
      _setKubeconfigLoader(null);
      _setKubeconfigCacheKeyFn(null);
      for (const key of ENV_KEYS) {
        if (savedEnv[key] === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = savedEnv[key];
        }
      }
    });

    it('MLFLOW_TRACKING_TOKEN takes precedence over MLFLOW_TRACKING_AUTH', async () => {
      process.env.MLFLOW_TRACKING_TOKEN = 'explicit-token';
      process.env.MLFLOW_TRACKING_AUTH = 'kubernetes';

      const provider = createAuthProvider({ trackingUri: 'http://mlflow:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer explicit-token');
    });

    it('kubernetes + username/password: basic auth ignored, k8s token used', async () => {
      // eslint-disable-next-line require-await, @typescript-eslint/require-await
      _setKubeconfigLoader(async () => ({ token: 'k8s-resolved-token' }));
      process.env.MLFLOW_TRACKING_USERNAME = 'admin';
      process.env.MLFLOW_TRACKING_PASSWORD = 'secret';
      process.env.MLFLOW_TRACKING_AUTH = 'kubernetes';

      const provider = createAuthProvider({ trackingUri: 'http://mlflow:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer k8s-resolved-token');
    });

    it('programmatic trackingServerToken skips kubernetes auth', async () => {
      process.env.MLFLOW_TRACKING_AUTH = 'kubernetes';

      const provider = createAuthProvider({
        trackingUri: 'http://mlflow:5000',
        trackingServerToken: 'code-token',
      });
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer code-token');
    });

    it('kubernetes-namespaced + explicit token: uses token but still auto-discovers workspace', async () => {
      // eslint-disable-next-line require-await, @typescript-eslint/require-await
      _setKubeconfigLoader(async () => ({}));
      const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mlflow-auth-prec-'));
      try {
        const namespacePath = path.join(tmpDir, 'namespace');
        fs.writeFileSync(namespacePath, 'discovered-ns');

        const provider = createKubernetesAuth(
          'http://mlflow:5000',
          true,
          path.join(tmpDir, 'no-token'),
          namespacePath,
          undefined,
          'Bearer explicit-token',
        );
        const headers = await provider.getHeadersProvider()();

        expect(headers['Authorization']).toBe('Bearer explicit-token');
        expect(headers['X-MLFLOW-WORKSPACE']).toBe('discovered-ns');
      } finally {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      }
    });

    it('kubernetes-namespaced + username/password: basic auth ignored, k8s auth + auto workspace', async () => {
      _setKubeconfigLoader(
        // eslint-disable-next-line require-await, @typescript-eslint/require-await
        async () => ({ token: 'k8s-resolved-token', namespace: 'discovered-ns' }),
      );
      process.env.MLFLOW_TRACKING_USERNAME = 'admin';
      process.env.MLFLOW_TRACKING_PASSWORD = 'secret';
      process.env.MLFLOW_TRACKING_AUTH = 'kubernetes-namespaced';

      const provider = createAuthProvider({ trackingUri: 'http://mlflow:5000' });
      const headers = await provider.getHeadersProvider()();

      expect(headers['Authorization']).toBe('Bearer k8s-resolved-token');
      expect(headers['X-MLFLOW-WORKSPACE']).toBe('discovered-ns');
    });
  });
});
