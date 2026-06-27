import { createAuthProvider } from '../../src/auth';

const ENV_KEYS = [
  'MLFLOW_TRACKING_TOKEN',
  'MLFLOW_TRACKING_USERNAME',
  'MLFLOW_TRACKING_PASSWORD',
  'MLFLOW_WORKSPACE',
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
});
