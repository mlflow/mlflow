import path from 'path';
import os from 'os';
import { initializeSDK } from './provider';
import { AuthProvider, createAuthProvider, isDatabricksUri } from '../auth';

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
   * @deprecated For Databricks auth, prefer using environment variables or config profiles.
   * The auth system will automatically resolve credentials.
   */
  host?: string;

  /**
   * The Databricks token. If not provided, the token will be read from the Databricks config file.
   * @deprecated For Databricks auth, prefer using environment variables or config profiles.
   * The auth system will automatically resolve credentials.
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

  /**
   * Bearer token for OSS MLflow tracking servers.
   * Can also be set via MLFLOW_TRACKING_TOKEN environment variable.
   */
  trackingServerToken?: string;
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
 * Global authentication provider
 */
let globalAuthProvider: AuthProvider | null = null;

/**
 * Configure the MLflow tracing SDK with tracking location settings.
 * This must be called before using other tracing functions.
 *
 * ## Authentication
 *
 * ### Databricks (trackingUri = "databricks" or "databricks://profile")
 *
 * Authentication is handled automatically by the Databricks SDK. It supports:
 * - Personal Access Tokens (DATABRICKS_TOKEN env var)
 * - OAuth M2M service principals (DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET)
 * - Azure CLI / Azure MSI / Azure Client Secret
 * - Google Cloud credentials
 * - Config file profiles (~/.databrickscfg)
 *
 * For Databricks Apps, credentials are automatically injected by the runtime.
 *
 * ### OSS MLflow (trackingUri = HTTP URL)
 *
 * Authentication is resolved in priority order:
 * 1. Basic Auth (trackingServerUsername/Password or MLFLOW_TRACKING_USERNAME/PASSWORD)
 * 2. Bearer Token (trackingServerToken or MLFLOW_TRACKING_TOKEN)
 * 3. No Auth
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
 * // Option 5: Override with explicit host/token (backwards compatibility)
 * init({
 *   trackingUri: "databricks",
 *   experimentId: "123456789",
 *   host: "https://my-workspace.databricks.com",
 *   databricksToken: "my-token"
 * });
 *
 * // Option 6: OSS MLflow with basic auth
 * init({
 *   trackingUri: "http://localhost:5000",
 *   experimentId: "123456789",
 *   trackingServerUsername: "user",
 *   trackingServerPassword: "pass"
 * });
 *
 * // Option 7: OSS MLflow with bearer token
 * init({
 *   trackingUri: "http://localhost:5000",
 *   experimentId: "123456789",
 *   trackingServerToken: "my-token"
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
      'An MLflow Tracking URI is required, please provide the trackingUri option to init, or set the MLFLOW_TRACKING_URI environment variable',
    );
  }

  if (!experimentId) {
    throw new Error(
      'An MLflow experiment ID is required, please provide the experimentId option to init, or set the MLFLOW_EXPERIMENT_ID environment variable',
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

  // Validate non-Databricks URIs
  if (!isDatabricksUri(trackingUri) && !isValidHttpUri(trackingUri)) {
    throw new Error(
      `Invalid trackingUri: '${trackingUri}'. Must be a valid HTTP or HTTPS URL, or 'databricks' / 'databricks://<profile>'.`,
    );
  }

  // Create the authentication provider - this is the single source of truth
  // for credential resolution. It handles env vars, config files, OAuth, etc.
  globalAuthProvider = createAuthProvider({
    trackingUri,
    host: config.host,
    databricksToken: config.databricksToken,
    databricksConfigPath,
    trackingServerUsername: config.trackingServerUsername,
    trackingServerPassword: config.trackingServerPassword,
    trackingServerToken: config.trackingServerToken,
  });

  // Build effective config, populating host and databricksToken from auth provider
  // for backwards compatibility with code that reads these properties directly
  const effectiveConfig: MLflowTracingConfig = {
    ...config,
    trackingUri,
    experimentId,
    databricksConfigPath,
    host: globalAuthProvider.getHost(),
    databricksToken: globalAuthProvider.getDatabricksToken(),
  };

  // Store the config
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
      'The MLflow Tracing client is not configured. Please call init() with host and experimentId before using tracing functions.',
    );
  }
  return globalConfig;
}

/**
 * Get the current authentication provider. Throws an error if not configured.
 * @returns The current authentication provider
 */
export function getAuthProvider(): AuthProvider {
  if (!globalAuthProvider) {
    throw new Error(
      'The MLflow Tracing client is not configured. Please call init() before using tracing functions.',
    );
  }
  return globalAuthProvider;
}

/**
 * Reset the global configuration. For testing purposes only.
 * @internal
 */
export function resetConfig(): void {
  globalConfig = null;
  globalAuthProvider = null;
}
