import path from 'path';
import fs from 'fs';
import os from 'os';
import { parse as parseIni } from 'ini';
import { initializeSDK } from './provider';

/**
 * Default Databricks profile name
 */
const DEFAULT_PROFILE = 'DEFAULT';

/**
 * Validate that a URI has a proper protocol (http or https)
 * @param uri The URI to validate
 * @returns true if valid, false otherwise
 */
function isValidHttpUri(uri: string): boolean {
  try {
    const url = new URL(uri);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

/**
 * Configuration options for the MLflow tracing SDK
 */
export interface MLflowTracingConfig {
  /**
   * The host where traces will be logged.
   * Can be:
   * - HTTP URI for MLflow trace server backend
   * - "databricks" (uses default profile)
   * - "databricks://profile" (uses specific profile)
   */
  trackingUri: string;

  /**
   * The experiment ID where traces will be logged
   */
  experimentId: string;

  /**
   * The location of the Databricks config file, default to ~/.databrickscfg
   */
  databricksConfigPath?: string;

  /**
   * The Databricks host. If not provided, the host will be read from the Databricks config file.
   */
  host?: string;

  /**
   * The Databricks token. If not provided, the token will be read from the Databricks config file.
   */
  databricksToken?: string;

  /**
   * The tracking server username for basic auth.
   */
  trackingServerUsername?: string;

  /**
   * The tracking server password for basic auth.
   */
  trackingServerPassword?: string;
}

/**
 * Global configuration state
 */
let globalConfig: MLflowTracingConfig | null = null;

/**
 * Configure the MLflow tracing SDK with tracking location settings.
 * This must be called before using other tracing functions.
 *
 * @param config Configuration object with trackingUri and experimentId
 *
 * @example
 * ```typescript
 * import { init, withSpan } from 'mlflow-tracing-ts';
 *
 * // Option 1: Use MLflow Tracking Server
 * init({
 *   trackingUri: "http://localhost:5000",
 *   experimentId: "123456789"
 * });
 *
 * // Option 2: Use default Databricks profile from ~/.databrickscfg
 * init({
 *   trackingUri: "databricks",
 *   experimentId: "123456789"
 * });
 *
 * // Option 3: Use specific Databricks profile
 * init({
 *   trackingUri: "databricks://my-profile",
 *   experimentId: "123456789"
 * });
 *
 * // Option 4: Custom config file path
 * init({
 *   trackingUri: "databricks",
 *   experimentId: "123456789",
 *   databricksConfigPath: "/path/to/my/databrickscfg"
 * });
 *
 * // Option 5: Override with explicit host/token (bypasses config file)
 * init({
 *   trackingUri: "databricks",
 *   experimentId: "123456789",
 *   host: "https://my-workspace.databricks.com",
 *   databricksToken: "my-token"
 * });
 *
 * // Now you can use tracing functions
 * function add(a: number, b: number) {
 *   return withSpan(
 *     { name: 'add', inputs: { a, b } },
 *     (span) => {
 *       const result = a + b;
 *       span.setOutputs({ result });
 *       return result;
 *     }
 *   );
 * }
 * ```
 */
export function init(config: MLflowTracingConfig): void {
  if (!config.trackingUri) {
    throw new Error('trackingUri is required in configuration');
  }

  if (!config.experimentId) {
    throw new Error('experimentId is required in configuration');
  }

  if (typeof config.trackingUri !== 'string') {
    throw new Error('trackingUri must be a string');
  }

  if (typeof config.experimentId !== 'string') {
    throw new Error('experimentId must be a string');
  }

  // Set default Databricks config path if not provided
  if (!config.databricksConfigPath) {
    config.databricksConfigPath = path.join(os.homedir(), '.databrickscfg');
  }

  if (config.trackingUri === 'databricks' || config.trackingUri.startsWith('databricks://')) {
    // Parse host and token from Databricks config file or environment variables for databricks URIs
    if (!config.host || !config.databricksToken) {
      // First try environment variables
      const envHost = process.env.DATABRICKS_HOST;
      const envToken = process.env.DATABRICKS_TOKEN;

      if (envHost && envToken) {
        if (!config.host) {
          config.host = envHost;
        }
        if (!config.databricksToken) {
          config.databricksToken = envToken;
        }
      } else {
        // Fall back to config file
        // Determine profile name from trackingUri
        let profile = DEFAULT_PROFILE;
        if (config.trackingUri.startsWith('databricks://')) {
          const profilePart = config.trackingUri.slice(13); // Remove 'databricks://'
          if (profilePart && profilePart.length > 0) {
            profile = profilePart;
          }
        }

        try {
          const { host, token } = readDatabricksConfig(config.databricksConfigPath, profile);
          if (!config.host) {
            config.host = host;
          }
          if (!config.databricksToken) {
            config.databricksToken = token;
          }
        } catch (error) {
          throw new Error(
            `Failed to read Databricks configuration for profile '${profile}': ${(error as Error).message}. ` +
              `Make sure your ${config.databricksConfigPath} file exists and contains valid credentials, or set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.`
          );
        }
      }
    }
  } else {
    // For self-hosted MLflow tracking server, validate and use the trackingUri as the host
    if (!isValidHttpUri(config.trackingUri)) {
      throw new Error(
        `Invalid trackingUri: '${config.trackingUri}'. Must be a valid HTTP or HTTPS URL.`
      );
    }
    config.host = config.trackingUri;
  }

  globalConfig = { ...config };

  // Initialize SDK with new configuration
  initializeSDK();
}

/**
 * Get the current configuration. Throws an error if not configured.
 * @returns The current MLflow tracing configuration
 */
export function getConfig(): MLflowTracingConfig {
  if (!globalConfig) {
    throw new Error(
      'The MLflow Tracing client is not configured. Please call init() with host and experimentId before using tracing functions.'
    );
  }
  return globalConfig;
}

/**
 * Read Databricks configuration from .databrickscfg file
 * @internal This function is exported only for testing purposes
 * @param configPath Path to the Databricks config file
 * @param profile Profile name to read (defaults to 'DEFAULT')
 * @returns Object containing host and token
 */
export function readDatabricksConfig(configPath: string, profile: string = DEFAULT_PROFILE) {
  try {
    if (!fs.existsSync(configPath)) {
      throw new Error(`Databricks config file not found at ${configPath}`);
    }

    const configContent = fs.readFileSync(configPath, 'utf8');
    const config = parseIni(configContent);

    if (!config[profile]) {
      throw new Error(`Profile '${profile}' not found in Databricks config file`);
    }

    const profileConfig = config[profile] as { host?: string; token?: string };

    if (!profileConfig.host) {
      throw new Error(`Host not found for profile '${profile}' in Databricks config file`);
    }

    if (!profileConfig.token) {
      throw new Error(`Token not found for profile '${profile}' in Databricks config file`);
    }

    return {
      host: profileConfig.host,
      token: profileConfig.token
    };
  } catch (error) {
    throw new Error(`Failed to read Databricks config: ${(error as Error).message}`);
  }
}
