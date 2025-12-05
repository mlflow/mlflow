import { init, getConfig, resetConfig, ensureInitialized } from '../../src/core/config';
import { DatabricksSdkAuthProvider } from '../../src/auth/providers/databricks-sdk';

// Mock the DatabricksSdkAuthProvider
jest.mock('../../src/auth/providers/databricks-sdk');
const MockDatabricksSdkAuthProvider = DatabricksSdkAuthProvider as jest.MockedClass<
  typeof DatabricksSdkAuthProvider
>;

// Mock the provider module to prevent actual SDK initialization
jest.mock('../../src/core/provider', () => ({
  initializeSDKAsync: jest.fn().mockResolvedValue(undefined)
}));

describe('Config', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    resetConfig();
    // Reset environment variables
    delete process.env.MLFLOW_TRACKING_URI;
    delete process.env.MLFLOW_EXPERIMENT_ID;
    delete process.env.DATABRICKS_HOST;
    delete process.env.DATABRICKS_TOKEN;
  });

  describe('init and getConfig', () => {
    describe('environment variable resolution', () => {
      it('should read tracking configuration from environment variables when not provided', () => {
        process.env.MLFLOW_TRACKING_URI = 'http://env-tracking-host:5000';
        process.env.MLFLOW_EXPERIMENT_ID = 'env-experiment-id';

        init({});

        const result = getConfig();
        expect(result.trackingUri).toBe('http://env-tracking-host:5000');
        expect(result.experimentId).toBe('env-experiment-id');
        expect(result.host).toBe('http://env-tracking-host:5000');
      });

      it('should throw an error when trackingUri is missing from both config and environment', () => {
        process.env.MLFLOW_EXPERIMENT_ID = 'env-experiment-id';

        expect(() => init({ experimentId: 'config-experiment-id' })).toThrow(
          'An MLflow Tracking URI is required, please provide the trackingUri option to init, or set the MLFLOW_TRACKING_URI environment variable'
        );
      });

      it('should throw an error when experimentId is missing from both config and environment', () => {
        process.env.MLFLOW_TRACKING_URI = 'http://env-tracking-host:5000';

        expect(() => init({ trackingUri: 'http://explicit-host:5000' })).toThrow(
          'An MLflow experiment ID is required, please provide the experimentId option to init, or set the MLFLOW_EXPERIMENT_ID environment variable'
        );
      });
    });

    it('should initialize with MLflow tracking server configuration', () => {
      const config = {
        trackingUri: 'http://localhost:5000',
        experimentId: '123456789'
      };

      init(config);

      const result = getConfig();
      expect(result.trackingUri).toBe('http://localhost:5000');
      expect(result.experimentId).toBe('123456789');
      expect(result.host).toBe('http://localhost:5000');
    });

    it('should throw error if trackingUri is missing', () => {
      const config = {
        trackingUri: '',
        experimentId: '123456789'
      };

      expect(() => init(config)).toThrow(
        'An MLflow Tracking URI is required, please provide the trackingUri option to init, or set the MLFLOW_TRACKING_URI environment variable'
      );
    });

    it('should throw error if experimentId is missing', () => {
      const config = {
        trackingUri: 'http://localhost:5000',
        experimentId: ''
      };

      expect(() => init(config)).toThrow(
        'An MLflow experiment ID is required, please provide the experimentId option to init, or set the MLFLOW_EXPERIMENT_ID environment variable'
      );
    });

    it('should throw error if trackingUri is not a string', () => {
      const config = {
        trackingUri: 123 as any,
        experimentId: '123456789'
      };

      expect(() => init(config)).toThrow('trackingUri must be a string');
    });

    it('should throw error if experimentId is not a string', () => {
      const config = {
        trackingUri: 'http://localhost:5000',
        experimentId: 123 as any
      };

      expect(() => init(config)).toThrow('experimentId must be a string');
    });

    it('should throw error for malformed trackingUri', () => {
      const config = {
        trackingUri: 'not-a-valid-uri',
        experimentId: '123456789'
      };

      expect(() => init(config)).toThrow(
        "Invalid trackingUri: 'not-a-valid-uri'. Must be a valid HTTP or HTTPS URL."
      );
    });

    describe('Databricks configuration', () => {
      beforeEach(() => {
        MockDatabricksSdkAuthProvider.mockImplementation(
          () =>
            ({
              authenticate: jest.fn().mockResolvedValue({ authorizationHeader: 'Bearer mock' }),
              getHost: jest.fn().mockResolvedValue('https://mock-workspace.databricks.com')
            }) as any
        );
      });

      it('should create DatabricksSdkAuthProvider for "databricks" tracking URI', () => {
        init({
          trackingUri: 'databricks',
          experimentId: '123456789'
        });

        expect(MockDatabricksSdkAuthProvider).toHaveBeenCalledWith({
          profile: undefined,
          configFile: undefined,
          host: undefined,
          token: undefined,
          clientId: undefined,
          clientSecret: undefined
        });

        const result = getConfig();
        expect(result.authProvider).toBeDefined();
      });

      it('should extract profile from "databricks://profile" tracking URI', () => {
        init({
          trackingUri: 'databricks://my-profile',
          experimentId: '123456789'
        });

        expect(MockDatabricksSdkAuthProvider).toHaveBeenCalledWith(
          expect.objectContaining({
            profile: 'my-profile'
          })
        );
      });

      it('should handle empty profile as undefined (databricks://)', () => {
        init({
          trackingUri: 'databricks://',
          experimentId: '123456789'
        });

        expect(MockDatabricksSdkAuthProvider).toHaveBeenCalledWith(
          expect.objectContaining({
            profile: undefined
          })
        );
      });

      it('should pass databricksConfigPath as configFile', () => {
        init({
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: '/custom/path/.databrickscfg'
        });

        expect(MockDatabricksSdkAuthProvider).toHaveBeenCalledWith(
          expect.objectContaining({
            configFile: '/custom/path/.databrickscfg'
          })
        );
      });

      it('should pass explicit host and token to auth provider', () => {
        init({
          trackingUri: 'databricks',
          experimentId: '123456789',
          host: 'https://explicit-host.databricks.com',
          databricksToken: 'explicit-token'
        });

        expect(MockDatabricksSdkAuthProvider).toHaveBeenCalledWith(
          expect.objectContaining({
            host: 'https://explicit-host.databricks.com',
            token: 'explicit-token'
          })
        );
      });

      it('should pass OAuth credentials to auth provider', () => {
        init({
          trackingUri: 'databricks',
          experimentId: '123456789',
          clientId: 'my-client-id',
          clientSecret: 'my-client-secret'
        });

        expect(MockDatabricksSdkAuthProvider).toHaveBeenCalledWith(
          expect.objectContaining({
            clientId: 'my-client-id',
            clientSecret: 'my-client-secret'
          })
        );
      });

      it('should not create auth provider if one is already provided', () => {
        const customAuthProvider = {
          authenticate: jest.fn().mockResolvedValue({ authorizationHeader: 'Custom' })
        };

        init({
          trackingUri: 'databricks',
          experimentId: '123456789',
          authProvider: customAuthProvider as any
        });

        expect(MockDatabricksSdkAuthProvider).not.toHaveBeenCalled();

        const result = getConfig();
        expect(result.authProvider).toBe(customAuthProvider);
      });

      it('should not set host for Databricks URIs (resolved lazily)', () => {
        init({
          trackingUri: 'databricks',
          experimentId: '123456789'
        });

        const result = getConfig();
        // Host is not set immediately for Databricks URIs - it's resolved lazily
        expect(result.host).toBeUndefined();
      });
    });
  });

  describe('ensureInitialized', () => {
    it('should resolve immediately if no init was called', async () => {
      // When init hasn't been called, there's no promise to wait for
      await expect(ensureInitialized()).resolves.toBeUndefined();
    });

    it('should wait for initialization to complete', async () => {
      init({
        trackingUri: 'http://localhost:5000',
        experimentId: '123'
      });

      // Should not throw
      await expect(ensureInitialized()).resolves.toBeUndefined();
    });
  });

  describe('resetConfig', () => {
    it('should clear global config', () => {
      init({
        trackingUri: 'http://localhost:5000',
        experimentId: '123'
      });

      resetConfig();

      expect(() => getConfig()).toThrow(
        'The MLflow Tracing client is not configured'
      );
    });
  });
});
