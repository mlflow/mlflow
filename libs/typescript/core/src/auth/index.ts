/**
 * MLflow TypeScript SDK Authentication Module
 *
 * This module provides authentication for MLflow tracking servers:
 *
 * ## Databricks-hosted MLflow
 *
 * For `databricks://` URIs, authentication is delegated entirely to the
 * Databricks SDK which supports multiple authentication methods:
 * - Personal Access Tokens (DATABRICKS_TOKEN)
 * - OAuth M2M / Service Principals (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
 * - Azure CLI, Azure MSI, Azure Client Secret
 * - Google Cloud credentials
 * - Config file profiles (~/.databrickscfg)
 *
 * Host resolution order:
 * 1. Explicit `host` option
 * 2. DATABRICKS_HOST environment variable
 * 3. Host from ~/.databrickscfg (for the specified profile)
 *
 * ## Self-hosted MLflow (OSS)
 *
 * For HTTP(S) URIs, authentication is resolved in order:
 * 1. Basic Auth (MLFLOW_TRACKING_USERNAME + MLFLOW_TRACKING_PASSWORD)
 * 2. Bearer Token (MLFLOW_TRACKING_TOKEN)
 * 3. No authentication
 *
 * @module auth
 */

import fs from 'fs';
import os from 'os';
import path from 'path';
import { Config } from '@databricks/sdk-experimental';

/**
 * Function that provides HTTP headers for authenticated requests.
 * Called before each request to support token refresh for OAuth.
 */
export type HeadersProvider = () => Promise<Record<string, string>>;

/**
 * Authentication provider interface used by MLflow clients.
 * Provides methods to get the host URL and authentication headers.
 */
export interface AuthProvider {
  /** Get the host URL for API requests */
  getHost(): string;

  /** Get the function that provides authentication headers */
  getHeadersProvider(): HeadersProvider;

  /**
   * Get the Databricks token if available (for backwards compatibility).
   * Returns undefined for non-Databricks or OAuth-only authentication.
   */
  getDatabricksToken(): string | undefined;
}

/**
 * Configuration options for authentication resolution.
 */
export interface AuthOptions {
  /** Tracking URI: "databricks", "databricks://profile", or HTTP(S) URL */
  trackingUri: string;

  /** Explicit host override (takes precedence over environment) */
  host?: string;

  /** Explicit Databricks token override */
  databricksToken?: string;

  /** Path to Databricks config file (default: ~/.databrickscfg) */
  databricksConfigPath?: string;

  /** Basic auth username for OSS MLflow */
  trackingServerUsername?: string;

  /** Basic auth password for OSS MLflow */
  trackingServerPassword?: string;

  /** Bearer token for OSS MLflow */
  trackingServerToken?: string;
}

/**
 * Check if a tracking URI targets Databricks.
 */
export function isDatabricksUri(trackingUri: string): boolean {
  return trackingUri === 'databricks' || trackingUri.startsWith('databricks://');
}

/**
 * Parse profile name from a Databricks tracking URI.
 *
 * @example
 * parseDatabricksProfile('databricks') // returns undefined (use DEFAULT)
 * parseDatabricksProfile('databricks://my-profile') // returns 'my-profile'
 */
export function parseDatabricksProfile(trackingUri: string): string | undefined {
  if (trackingUri === 'databricks') {
    return undefined;
  }
  const match = trackingUri.match(/^databricks:\/\/(.+)$/);
  return match?.[1] || undefined;
}

/**
 * Create an authentication provider for the given configuration.
 *
 * This is the main entry point for authentication. It returns an AuthProvider
 * with methods to get the host URL and authentication headers.
 *
 * @param options - Authentication configuration options
 * @returns AuthProvider for making authenticated requests
 * @throws Error if required configuration is missing
 */
export function createAuthProvider(options: AuthOptions): AuthProvider {
  if (isDatabricksUri(options.trackingUri)) {
    return createDatabricksAuth(options);
  }
  return createOssAuth(options);
}

/**
 * Read the host value from a Databricks config file for a specific profile.
 *
 * The config file uses INI format:
 * ```
 * [DEFAULT]
 * host = https://my-workspace.databricks.com
 * token = dapi123...
 *
 * [profile-name]
 * host = https://other-workspace.databricks.com
 * ```
 *
 * @param configPath - Path to the config file
 * @param profile - Profile name (undefined for DEFAULT)
 * @returns The host value or undefined if not found
 */
function readHostFromConfigFile(configPath: string, profile?: string): string | undefined {
  try {
    if (!fs.existsSync(configPath)) {
      return undefined;
    }

    const content = fs.readFileSync(configPath, 'utf-8');
    const targetSection = profile || 'DEFAULT';

    // Simple INI parser - find the section and extract host
    const lines = content.split('\n');
    let inTargetSection = false;

    for (const line of lines) {
      const trimmed = line.trim();

      // Check for section header
      const sectionMatch = trimmed.match(/^\[(.+)\]$/);
      if (sectionMatch) {
        inTargetSection = sectionMatch[1] === targetSection;
        continue;
      }

      // Look for host in current section
      if (inTargetSection) {
        const hostMatch = trimmed.match(/^host\s*=\s*(.+)$/);
        if (hostMatch) {
          return hostMatch[1].trim();
        }
      }
    }

    return undefined;
  } catch {
    return undefined;
  }
}

/**
 * Create authentication for Databricks-hosted MLflow.
 *
 * Host resolution order:
 * 1. Explicit `host` option
 * 2. DATABRICKS_HOST environment variable
 * 3. Host from config file (~/.databrickscfg)
 *
 * All credential resolution is delegated to the Databricks SDK.
 * The SDK handles env vars, config files, OAuth, and cloud-specific auth.
 */
function createDatabricksAuth(options: AuthOptions): AuthProvider {
  const profile = parseDatabricksProfile(options.trackingUri);
  const configPath = options.databricksConfigPath ?? path.join(os.homedir(), '.databrickscfg');

  // Resolve host - must be available synchronously
  // Priority: explicit option > env var > config file
  const host =
    options.host || process.env.DATABRICKS_HOST || readHostFromConfigFile(configPath, profile);

  if (!host) {
    throw new Error(
      'Databricks host not found. Please either:\n' +
        '  1. Set the DATABRICKS_HOST environment variable, or\n' +
        `  2. Add a host to your config file at ${configPath}`
    );
  }

  // Create Databricks SDK Config - it handles all credential resolution
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
  const databricksConfig = new Config({
    host: options.host,
    token: options.databricksToken,
    profile,
    configFile: options.databricksConfigPath
  });

  // Headers provider delegates entirely to the SDK
  const headersProvider: HeadersProvider = async () => {
    // SDK resolves credentials from env vars, config file, OAuth, etc.
    // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
    await databricksConfig.ensureResolved();

    const headers = new Headers();
    headers.set('Content-Type', 'application/json');

    // SDK adds appropriate auth headers (Bearer token, OAuth, etc.)
    // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
    await databricksConfig.authenticate(headers);

    // Convert Headers to plain object
    const result: Record<string, string> = {};
    headers.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  };

  // Token for backwards compatibility (may be undefined for OAuth)
  const databricksToken = options.databricksToken || process.env.DATABRICKS_TOKEN;

  return {
    getHost: () => host,
    getHeadersProvider: () => headersProvider,
    getDatabricksToken: () => databricksToken
  };
}

/**
 * Create authentication for self-hosted (OSS) MLflow.
 *
 * Resolution order:
 * 1. Explicit options (username/password or token)
 * 2. Environment variables (MLFLOW_TRACKING_*)
 * 3. No authentication
 */
function createOssAuth(options: AuthOptions): AuthProvider {
  const host = options.trackingUri;

  // Resolve credentials from options or environment
  const username = options.trackingServerUsername || process.env.MLFLOW_TRACKING_USERNAME;
  const password = options.trackingServerPassword || process.env.MLFLOW_TRACKING_PASSWORD;
  const token = options.trackingServerToken || process.env.MLFLOW_TRACKING_TOKEN;

  // Pre-compute auth header since credentials don't change
  let authHeader: string | undefined;
  if (username && password) {
    const encoded = Buffer.from(`${username}:${password}`).toString('base64');
    authHeader = `Basic ${encoded}`;
  } else if (token) {
    authHeader = `Bearer ${token}`;
  }

  // Headers provider for OSS MLflow
  // eslint-disable-next-line require-await, @typescript-eslint/require-await
  const headersProvider: HeadersProvider = async () => {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
    return headers;
  };

  return {
    getHost: () => host,
    getHeadersProvider: () => headersProvider,
    getDatabricksToken: () => undefined
  };
}
