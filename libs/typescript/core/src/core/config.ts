import { AuthProvider } from '../auth/types';
import { DatabricksSdkAuthProvider } from '../auth/providers';
import { initializeSDKAsync } from './provider';

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
   * The location of the Databricks config file.
   * Defaults to ~/.databrickscfg if not specified.
   * Forwarded to Databricks SDK's configFile option.
   */
  databricksConfigPath?: string;

  /**
   * The Databricks host. If not provided, the host will be resolved
   * automatically from environment variables or config file.
   */
  host?: string;

  /**
   * Authentication provider for API requests.
   * If not provided for Databricks URIs, a DatabricksSdkAuthProvider
   * will be created automatically with credential auto-discovery.
   */
  authProvider?: AuthProvider;

  /**
   * The Databricks token. If not provided, the token will be discovered
   * automatically from environment variables or config file.
   */
  databricksToken?: string;

  /**
   * OAuth client ID for service principal authentication (Databricks Apps).
   * Used with clientSecret for OAuth M2M flow.
   */
  clientId?: string;

  /**
   * OAuth client secret for service principal authentication (Databricks Apps).
   * Used with clientId for OAuth M2M flow.
   */
  clientSecret?: string;

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
 * Promise for async initialization. Used to ensure SDK is ready before exporting traces.
 */
let initPromise: Promise<void> | null = null;

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
 * // SDK will auto-discover credentials from env vars or config file
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
 * // Option 5: Override with explicit host/token
 * init({
 *   trackingUri: "databricks",
 *   experimentId: "123456789",
 *   host: "https://my-workspace.databricks.com",
 *   databricksToken: "my-token"
 * });
 *
 * // Option 6: OAuth M2M for Databricks Apps
 * // (SDK auto-discovers from DATABRICKS_CLIENT_ID/SECRET env vars)
 * init({
 *   trackingUri: "databricks",
 *   experimentId: "123456789"
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

  const effectiveConfig: MLflowTracingConfig = {
    ...config,
    trackingUri,
    experimentId
  };

  if (
    effectiveConfig.trackingUri === 'databricks' ||
    effectiveConfig.trackingUri?.startsWith('databricks://')
  ) {
    // Databricks tracking URI - use DatabricksSdkAuthProvider for automatic credential discovery
    // Extract profile from URI if specified (e.g., "databricks://my-profile" -> "my-profile")
    let profile: string | undefined;
    if (effectiveConfig.trackingUri.startsWith('databricks://')) {
      const profilePart = effectiveConfig.trackingUri.slice('databricks://'.length);
      if (profilePart) {
        profile = profilePart;
      }
    }

    // If user didn't provide an auth provider, create one with auto-discovery
    if (!effectiveConfig.authProvider) {
      effectiveConfig.authProvider = new DatabricksSdkAuthProvider({
        profile,
        configFile: config.databricksConfigPath,
        host: config.host,
        token: config.databricksToken,
        clientId: config.clientId,
        clientSecret: config.clientSecret
      });
    }

    // Host will be resolved asynchronously from the auth provider
    // Don't set it here - let initializeSDKAsync() resolve it
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

  // Start async initialization (fire-and-forget from caller's perspective)
  // This keeps init() synchronous for backwards compatibility
  initPromise = initializeSDKAsync();
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
 * Ensure that async initialization is complete.
 * This should be called before any operations that depend on the SDK being ready.
 * @internal
 */
export async function ensureInitialized(): Promise<void> {
  if (initPromise) {
    await initPromise;
  }
}

/**
 * Reset configuration state. Used for testing.
 * @internal
 */
export function resetConfig(): void {
  globalConfig = null;
  initPromise = null;
}
