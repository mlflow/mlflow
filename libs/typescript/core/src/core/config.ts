import path from 'path';
import fs from 'fs';
import os from 'os';
import { parse as parseIni } from 'ini';
import { initializeSDK } from './provider';
import { DatabricksAuthProvider, DatabricksAuthProviderOptions } from './databricks_auth';

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
   * Databricks OAuth client ID. When provided with client secret, an access token
   * will be fetched automatically instead of using a personal access token.
   */
  databricksClientId?: string;

  /**
   * Databricks OAuth client secret corresponding to the client ID.
   */
  databricksClientSecret?: string;

  /**
   * OAuth scopes to request when exchanging client credentials. Accepts either a
   * whitespace / comma separated string or an array of scopes.
   */
  databricksOauthScopes?: string | string[];

  /**
   * Optional override for the OAuth token endpoint. Defaults to `${host}/oidc/token`.
   */
  databricksOauthTokenEndpoint?: string;

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
let databricksAuthProvider: DatabricksAuthProvider | null = null;

export function getDatabricksAuthProvider(): DatabricksAuthProvider | null {
  return databricksAuthProvider;
}

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

    resolveDatabricksConfig(effectiveConfig, configPathToUse);
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

  databricksAuthProvider = createDatabricksAuthProvider(globalConfig);

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
export interface DatabricksProfileConfig {
  host: string;
  token?: string;
  clientId?: string;
  clientSecret?: string;
  oauthScopes?: string[];
  oauthTokenEndpoint?: string;
}

export function readDatabricksConfig(
  configPath: string,
  profile: string = DEFAULT_PROFILE
): DatabricksProfileConfig {
  try {
    if (!fs.existsSync(configPath)) {
      throw new Error(`Databricks config file not found at ${configPath}`);
    }

    const configContent = fs.readFileSync(configPath, 'utf8');
    const config = parseIni(configContent);

    const rawProfileConfig = config[profile];
    if (!rawProfileConfig) {
      throw new Error(`Profile '${profile}' not found in Databricks config file`);
    }

    const profileConfig = rawProfileConfig as {
      host?: string;
      token?: string;
      client_id?: string;
      client_secret?: string;
      clientId?: string;
      clientSecret?: string;
      oauth_scopes?: string;
      oauthScopes?: string;
      oauth_token_endpoint?: string;
      oauthTokenEndpoint?: string;
    };

    if (!profileConfig.host) {
      throw new Error(`Host not found for profile '${profile}' in Databricks config file`);
    }

    const token = profileConfig.token;
    const clientId = profileConfig.client_id ?? profileConfig.clientId;
    const clientSecret = profileConfig.client_secret ?? profileConfig.clientSecret;
    const oauthScopesRaw = profileConfig.oauth_scopes ?? profileConfig.oauthScopes;
    const oauthTokenEndpoint =
      profileConfig.oauth_token_endpoint ?? profileConfig.oauthTokenEndpoint;

    if (!token && !(clientId && clientSecret)) {
      throw new Error(
        `Authentication details not found for profile '${profile}' in Databricks config file`
      );
    }

    if ((clientId && !clientSecret) || (!clientId && clientSecret)) {
      throw new Error(
        `Incomplete client credential configuration for profile '${profile}' in Databricks config file`
      );
    }

    return {
      host: profileConfig.host,
      token,
      clientId,
      clientSecret,
      oauthScopes: normalizeScopes(oauthScopesRaw),
      oauthTokenEndpoint
    };
  } catch (error) {
    throw new Error(`Failed to read Databricks config: ${(error as Error).message}`);
  }
}

function normalizeScopes(scopes?: string | string[]): string[] | undefined {
  if (!scopes) {
    return undefined;
  }
  if (Array.isArray(scopes)) {
    return scopes.filter((scope) => !!scope);
  }
  return scopes
    .split(/[\s,]+/)
    .map((scope) => scope.trim())
    .filter((scope) => scope.length > 0);
}

function resolveDatabricksConfig(
  effectiveConfig: MLflowTracingConfig,
  configPathToUse?: string
): void {
  const envHost = process.env.DATABRICKS_HOST;
  const envToken = process.env.DATABRICKS_TOKEN;
  const envClientId = process.env.DATABRICKS_CLIENT_ID;
  const envClientSecret = process.env.DATABRICKS_CLIENT_SECRET;
  const envScopes = process.env.DATABRICKS_OAUTH_SCOPES;
  const envTokenEndpoint = process.env.DATABRICKS_OAUTH_TOKEN_ENDPOINT;

  if (envHost && !effectiveConfig.host) {
    effectiveConfig.host = envHost;
  }
  if (envToken && !effectiveConfig.databricksToken) {
    effectiveConfig.databricksToken = envToken;
  }
  if (envClientId && !effectiveConfig.databricksClientId) {
    effectiveConfig.databricksClientId = envClientId;
  }
  if (envClientSecret && !effectiveConfig.databricksClientSecret) {
    effectiveConfig.databricksClientSecret = envClientSecret;
  }
  if (envScopes && !effectiveConfig.databricksOauthScopes) {
    effectiveConfig.databricksOauthScopes = normalizeScopes(envScopes);
  }
  if (envTokenEndpoint && !effectiveConfig.databricksOauthTokenEndpoint) {
    effectiveConfig.databricksOauthTokenEndpoint = envTokenEndpoint;
  }

  if (!effectiveConfig.host || !hasAuthConfigured(effectiveConfig)) {
    // Determine profile name from trackingUri
    let profile = DEFAULT_PROFILE;
    if (effectiveConfig.trackingUri?.startsWith('databricks://')) {
      const profilePart = effectiveConfig.trackingUri.slice('databricks://'.length);
      if (profilePart) {
        profile = profilePart;
      }
    }

    try {
      const resolved = readDatabricksConfig(configPathToUse ?? '', profile);
      if (!effectiveConfig.host) {
        effectiveConfig.host = resolved.host;
      }
      if (!effectiveConfig.databricksToken && resolved.token) {
        effectiveConfig.databricksToken = resolved.token;
      }
      if (!effectiveConfig.databricksClientId && resolved.clientId) {
        effectiveConfig.databricksClientId = resolved.clientId;
      }
      if (!effectiveConfig.databricksClientSecret && resolved.clientSecret) {
        effectiveConfig.databricksClientSecret = resolved.clientSecret;
      }
      if (!effectiveConfig.databricksOauthScopes && resolved.oauthScopes) {
        effectiveConfig.databricksOauthScopes = resolved.oauthScopes;
      }
      if (!effectiveConfig.databricksOauthTokenEndpoint && resolved.oauthTokenEndpoint) {
        effectiveConfig.databricksOauthTokenEndpoint = resolved.oauthTokenEndpoint;
      }
    } catch (error) {
      throw new Error(
        `Failed to read Databricks configuration for profile '${profile}': ${(error as Error).message}. ` +
          `Make sure your ${configPathToUse} file exists and contains valid credentials, or set DATABRICKS_HOST and relevant authentication environment variables.`
      );
    }
  }

  if (!effectiveConfig.host) {
    throw new Error(
      'Failed to determine Databricks host. Please set host explicitly, configure it in the Databricks config file, or set the DATABRICKS_HOST environment variable.'
    );
  }

  if (!hasAuthConfigured(effectiveConfig)) {
    throw new Error(
      'Databricks authentication is not configured. Provide a personal access token or client credentials via init options, environment variables, or the Databricks config file.'
    );
  }

  if (hasPartialClientCredentials(effectiveConfig)) {
    throw new Error(
      'Both databricksClientId and databricksClientSecret must be provided when using client credential authentication.'
    );
  }
}

function hasAuthConfigured(config: MLflowTracingConfig): boolean {
  return (
    Boolean(config.databricksToken) ||
    Boolean(config.databricksClientId && config.databricksClientSecret)
  );
}

function hasPartialClientCredentials(config: MLflowTracingConfig): boolean {
  return Boolean(
    (config.databricksClientId && !config.databricksClientSecret) ||
      (!config.databricksClientId && config.databricksClientSecret)
  );
}

function createDatabricksAuthProvider(
  config: MLflowTracingConfig
): DatabricksAuthProvider | null {
  if (!config.host) {
    return null;
  }

  if (
    config.trackingUri !== 'databricks' &&
    !config.trackingUri?.startsWith('databricks://')
  ) {
    return null;
  }

  const scopes = Array.isArray(config.databricksOauthScopes)
    ? config.databricksOauthScopes
    : normalizeScopes(config.databricksOauthScopes);

  const options: DatabricksAuthProviderOptions = {
    host: config.host,
    initialToken: config.databricksToken,
    clientId: config.databricksClientId,
    clientSecret: config.databricksClientSecret,
    scopes,
    tokenEndpoint: config.databricksOauthTokenEndpoint
  };

  return new DatabricksAuthProvider(options);
}
