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
 * Initialization options for the MLflow tracing SDK. The trackingUri and experimentId
 * can be omitted and will be resolved from environment variables when available. Since
 * this is used on the client side, we need to make sure the trackingUri and experimentId
 * are required.
 */
export type MLflowTracingInitOptions = Partial<MLflowTracingConfig>;

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
export function init(config: MLflowTracingInitOptions): void {
  const trackingUri = config.trackingUri ?? process.env.MLFLOW_TRACKING_URI;
  const experimentId = config.experimentId ?? process.env.MLFLOW_EXPERIMENT_ID;

  if (!trackingUri) {
    throw new Error(
      'An MLflow Tracking URI is required, please provide the trackingUri option to init, or set the MLFLOW_TRACKING_URI environment variable'
    );
  }

  if (!experimentId) {
    throw new Error(
      'An MLflow experiment ID is required, please provide the experimentId option to init, or set the MLFLOW_EXPERIMENT_ID environment variable'
    );
  }

  if (typeof trackingUri !== 'string') {
    throw new Error('trackingUri must be a string');
  }

  if (typeof experimentId !== 'string') {
    throw new Error('experimentId must be a string');
  }

  const databricksConfigPath =
    config.databricksConfigPath ?? path.join(os.homedir(), '.databrickscfg');

  const effectiveConfig: MLflowTracingConfig = {
    ...config,
    trackingUri,
    experimentId,
    databricksConfigPath
  };

  if (
    effectiveConfig.trackingUri === 'databricks' ||
    effectiveConfig.trackingUri?.startsWith('databricks://')
  ) {
    const configPathToUse = effectiveConfig.databricksConfigPath;

    if (!effectiveConfig.host || !effectiveConfig.databricksToken) {
      const envHost = process.env.DATABRICKS_HOST;
      const envToken = process.env.DATABRICKS_TOKEN;

      if (envHost && envToken) {
        if (!effectiveConfig.host) {
          effectiveConfig.host = envHost;
        }
        if (!effectiveConfig.databricksToken) {
          effectiveConfig.databricksToken = envToken;
        }
      } else {
        // Fall back to config file
        // Determine profile name from trackingUri
        let profile = DEFAULT_PROFILE;
        if (effectiveConfig.trackingUri?.startsWith('databricks://')) {
          const profilePart = effectiveConfig.trackingUri.slice('databricks://'.length);
          if (profilePart) {
            profile = profilePart;
          }
        }

        try {
          const { host, token } = readDatabricksConfig(configPathToUse ?? '', profile);
          if (!effectiveConfig.host) {
            effectiveConfig.host = host;
          }
          if (!effectiveConfig.databricksToken) {
            effectiveConfig.databricksToken = token;
          }
        } catch (error) {
          throw new Error(
            `Failed to read Databricks configuration for profile '${profile}': ${(error as Error).message}. ` +
              `Make sure your ${configPathToUse} file exists and contains valid credentials, or set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.`
          );
        }
      }
    }
  } else {
    // For self-hosted MLflow tracking server, validate and use the trackingUri as the host
    if (!isValidHttpUri(effectiveConfig.trackingUri ?? '')) {
      throw new Error(
        `Invalid trackingUri: '${effectiveConfig.trackingUri}'. Must be a valid HTTP or HTTPS URL.`
      );
    }
    effectiveConfig.host = effectiveConfig.trackingUri;
  }

  globalConfig = { ...effectiveConfig };

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
