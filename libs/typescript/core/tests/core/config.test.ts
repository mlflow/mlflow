import fs from 'fs';
import os from 'os';
import path from 'path';
import { init, getConfig } from '../../src/core/config';

describe('Config', () => {
  describe('init and getConfig', () => {
    describe('environment variable resolution', () => {
      afterEach(() => {
        delete process.env.MLFLOW_TRACKING_URI;
        delete process.env.MLFLOW_EXPERIMENT_ID;
      });

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
        "Invalid trackingUri: 'not-a-valid-uri'. Must be a valid HTTP or HTTPS URL, or 'databricks' / 'databricks://<profile>'."
      );
    });

    it.skip('should throw error if getConfig is called without init', () => {
      // Skip this test as it interferes with other tests due to module state
      expect(() => getConfig()).toThrow(
        'The MLflow Tracing client is not configured. Please call init() with host and experimentId before using tracing functions.'
      );
    });

    describe('Databricks configuration', () => {
      const tempDir = path.join(os.tmpdir(), 'mlflow-databricks-test-' + Date.now());
      const configPath = path.join(tempDir, '.databrickscfg');

      beforeEach(() => {
        fs.mkdirSync(tempDir, { recursive: true });
      });

      afterEach(() => {
        fs.rmSync(tempDir, { recursive: true, force: true });
        delete process.env.DATABRICKS_HOST;
        delete process.env.DATABRICKS_TOKEN;
      });

      it('should throw error if host not found anywhere', () => {
        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: '/nonexistent/path/.databrickscfg'
        };

        expect(() => init(config)).toThrow('Databricks host not found');
      });

      it('should read host from config file when env var not set', () => {
        const configContent = `[DEFAULT]
host = https://config-workspace.databricks.com
token = dapi123456789abcdef`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://config-workspace.databricks.com');
      });

      it('should read host from specific profile in config file', () => {
        const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com
token = default-token

[dev]
host = https://dev-workspace.databricks.com
token = dev-token`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks://dev',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://dev-workspace.databricks.com');
      });

      it('should use DATABRICKS_HOST from environment', () => {
        process.env.DATABRICKS_HOST = 'https://env-workspace.databricks.com';
        process.env.DATABRICKS_TOKEN = 'env-token';

        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789'
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://env-workspace.databricks.com');
        expect(result.databricksToken).toBe('env-token');
      });

      it('should use explicit host/token over environment', () => {
        process.env.DATABRICKS_HOST = 'https://env-workspace.databricks.com';
        process.env.DATABRICKS_TOKEN = 'env-token';

        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          host: 'https://override-workspace.databricks.com',
          databricksToken: 'override-token'
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://override-workspace.databricks.com');
        expect(result.databricksToken).toBe('override-token');
      });

      it('should prefer env var over config file for host', () => {
        process.env.DATABRICKS_HOST = 'https://env-workspace.databricks.com';

        const configContent = `[DEFAULT]
host = https://config-workspace.databricks.com
token = dapi123456789abcdef`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        // Env var takes precedence over config file
        expect(result.host).toBe('https://env-workspace.databricks.com');
      });

      it('should handle empty profile name in URI', () => {
        const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com
token = dapi123456789abcdef`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks://',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://default-workspace.databricks.com');
      });
    });
  });
});
