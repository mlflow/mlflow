import fs from 'fs';
import os from 'os';
import path from 'path';
import { init, getConfig, readDatabricksConfig } from '../../src/core/config';

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
        "Invalid trackingUri: 'not-a-valid-uri'. Must be a valid HTTP or HTTPS URL."
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
      });

      it('should read Databricks config for default profile', () => {
        const configContent = `[DEFAULT]
host = https://my-workspace.databricks.com
token = dapi123456789abcdef

[dev]
host = https://dev-workspace.databricks.com
token = dapi987654321fedcba`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://my-workspace.databricks.com');
        expect(result.databricksToken).toBe('dapi123456789abcdef');
      });

      it('should read Databricks config for specific profile', () => {
        const configContent = `[DEFAULT]
host = https://my-workspace.databricks.com
token = dapi123456789abcdef

[dev]
host = https://dev-workspace.databricks.com
token = dapi987654321fedcba`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks://dev',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://dev-workspace.databricks.com');
        expect(result.databricksToken).toBe('dapi987654321fedcba');
      });

      it('should use explicit host/token over config file', () => {
        const configContent = `[DEFAULT]
host = https://my-workspace.databricks.com
token = dapi123456789abcdef`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: configPath,
          host: 'https://override-workspace.databricks.com',
          databricksToken: 'override-token'
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://override-workspace.databricks.com');
        expect(result.databricksToken).toBe('override-token');
      });

      it('should throw error if Databricks config file not found', () => {
        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: '/nonexistent/path/.databrickscfg'
        };

        expect(() => init(config)).toThrow(/Failed to read Databricks configuration/);
        expect(() => init(config)).toThrow(
          /Make sure your \/nonexistent\/path\/.databrickscfg file exists/
        );
      });

      it('should throw error if profile not found in config', () => {
        const configContent = `[DEFAULT]
host = https://my-workspace.databricks.com
token = dapi123456789abcdef`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks://nonexistent',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        expect(() => init(config)).toThrow(
          /Failed to read Databricks configuration for profile 'nonexistent'/
        );
      });

      it('should handle empty profile name as DEFAULT', () => {
        const configContent = `[DEFAULT]
host = https://my-workspace.databricks.com
token = dapi123456789abcdef`;

        fs.writeFileSync(configPath, configContent);

        const config = {
          trackingUri: 'databricks://',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://my-workspace.databricks.com');
        expect(result.databricksToken).toBe('dapi123456789abcdef');
      });

      it('should use environment variables over config file', () => {
        const configContent = `[DEFAULT]
host = https://config-workspace.databricks.com
token = config-token`;

        fs.writeFileSync(configPath, configContent);

        // Set environment variables
        process.env.DATABRICKS_HOST = 'https://env-workspace.databricks.com';
        process.env.DATABRICKS_TOKEN = 'env-token';

        const config = {
          trackingUri: 'databricks',
          experimentId: '123456789',
          databricksConfigPath: configPath
        };

        init(config);

        const result = getConfig();
        expect(result.host).toBe('https://env-workspace.databricks.com');
        expect(result.databricksToken).toBe('env-token');

        // Clean up environment variables
        delete process.env.DATABRICKS_HOST;
        delete process.env.DATABRICKS_TOKEN;
      });
    });
  });

  describe('readDatabricksConfig', () => {
    const tempDir = path.join(os.tmpdir(), 'mlflow-read-test-' + Date.now());
    const configPath = path.join(tempDir, '.databrickscfg');

    beforeEach(() => {
      fs.mkdirSync(tempDir, { recursive: true });
    });

    afterEach(() => {
      fs.rmSync(tempDir, { recursive: true, force: true });
    });

    it('should read DEFAULT profile by default', () => {
      const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com
token = default-token`;

      fs.writeFileSync(configPath, configContent);

      const result = readDatabricksConfig(configPath);
      expect(result.host).toBe('https://default-workspace.databricks.com');
      expect(result.token).toBe('default-token');
    });

    it('should read specific profile', () => {
      const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com
token = default-token

[production]
host = https://prod-workspace.databricks.com
token = prod-token`;

      fs.writeFileSync(configPath, configContent);

      const result = readDatabricksConfig(configPath, 'production');
      expect(result.host).toBe('https://prod-workspace.databricks.com');
      expect(result.token).toBe('prod-token');
    });

    it('should handle config with extra fields', () => {
      const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com
token = default-token
username = user@example.com
jobs-api-version = 2.1`;

      fs.writeFileSync(configPath, configContent);

      const result = readDatabricksConfig(configPath);
      expect(result.host).toBe('https://default-workspace.databricks.com');
      expect(result.token).toBe('default-token');
    });

    it('should throw error if config file not found', () => {
      expect(() => readDatabricksConfig('/nonexistent/path/.databrickscfg')).toThrow(
        'Failed to read Databricks config: Databricks config file not found at /nonexistent/path/.databrickscfg'
      );
    });

    it('should throw error if profile not found', () => {
      const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com
token = default-token`;

      fs.writeFileSync(configPath, configContent);

      expect(() => readDatabricksConfig(configPath, 'nonexistent')).toThrow(
        "Failed to read Databricks config: Profile 'nonexistent' not found in Databricks config file"
      );
    });

    it('should throw error if host missing in profile', () => {
      const configContent = `[DEFAULT]
token = default-token`;

      fs.writeFileSync(configPath, configContent);

      expect(() => readDatabricksConfig(configPath)).toThrow(
        "Failed to read Databricks config: Host not found for profile 'DEFAULT' in Databricks config file"
      );
    });

    it('should throw error if token missing in profile', () => {
      const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com`;

      fs.writeFileSync(configPath, configContent);

      expect(() => readDatabricksConfig(configPath)).toThrow(
        "Failed to read Databricks config: Token not found for profile 'DEFAULT' in Databricks config file"
      );
    });

    it('should handle malformed config file', () => {
      const configContent = `This is not a valid INI file
[missing closing bracket
host = value`;

      fs.writeFileSync(configPath, configContent);

      // The ini parser is permissive, so this might not throw, but profile won't be found
      expect(() => readDatabricksConfig(configPath, 'DEFAULT')).toThrow(
        /Failed to read Databricks config/
      );
    });

    it('should read Databricks config for multiple profiles', () => {
      const configContent = `[DEFAULT]
host = https://default-workspace.databricks.com
token = dapi123456789abcdef

[dev]
host = https://dev-workspace.databricks.com
token = dapidev123456789ab

[staging]
host = https://staging-workspace.databricks.com
token = dapistaging123456`;

      fs.writeFileSync(configPath, configContent);

      // Test DEFAULT profile
      let result = readDatabricksConfig(configPath);
      expect(result.host).toBe('https://default-workspace.databricks.com');
      expect(result.token).toBe('dapi123456789abcdef');

      // Test dev profile
      result = readDatabricksConfig(configPath, 'dev');
      expect(result.host).toBe('https://dev-workspace.databricks.com');
      expect(result.token).toBe('dapidev123456789ab');

      // Test staging profile
      result = readDatabricksConfig(configPath, 'staging');
      expect(result.host).toBe('https://staging-workspace.databricks.com');
      expect(result.token).toBe('dapistaging123456');
    });

    it('should handle Azure Databricks config format', () => {
      const configContent = `[azure]
host = https://adb-1234567890123456.7.azuredatabricks.net
token = dapiazure123456789abcdef`;

      fs.writeFileSync(configPath, configContent);

      const result = readDatabricksConfig(configPath, 'azure');
      expect(result.host).toBe('https://adb-1234567890123456.7.azuredatabricks.net');
      expect(result.token).toBe('dapiazure123456789abcdef');
    });

    it('should handle AWS Databricks config format', () => {
      const configContent = `[aws]
host = https://dbc-abcd1234-5678.cloud.databricks.com
token = dapiaws123456789abcdef`;

      fs.writeFileSync(configPath, configContent);

      const result = readDatabricksConfig(configPath, 'aws');
      expect(result.host).toBe('https://dbc-abcd1234-5678.cloud.databricks.com');
      expect(result.token).toBe('dapiaws123456789abcdef');
    });

    it('should handle GCP Databricks config format', () => {
      const configContent = `[gcp]
host = https://1234567890123456.7.gcp.databricks.com
token = dapigcp123456789abcdef`;

      fs.writeFileSync(configPath, configContent);

      const result = readDatabricksConfig(configPath, 'gcp');
      expect(result.host).toBe('https://1234567890123456.7.gcp.databricks.com');
      expect(result.token).toBe('dapigcp123456789abcdef');
    });
  });
});
