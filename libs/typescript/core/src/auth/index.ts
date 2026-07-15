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
 * 3. Kubernetes auth (MLFLOW_TRACKING_AUTH=kubernetes or kubernetes-namespaced)
 * 4. No authentication
 *
 * Kubernetes auth support:
 * - MLFLOW_TRACKING_AUTH=kubernetes: reads bearer token from the pod's ServiceAccount mount
 * - MLFLOW_TRACKING_AUTH=kubernetes-namespaced: same, plus sets X-MLFLOW-WORKSPACE from the namespace
 *
 * Workspace support (servers with workspaces enabled):
 * - MLFLOW_WORKSPACE: sets the `X-MLFLOW-WORKSPACE` header for workspace-scoped requests
 *
 * @module auth
 */

import fs from 'fs';
import os from 'os';
import path from 'path';
import { Config } from '@databricks/sdk-experimental';

const SA_TOKEN_PATH = '/var/run/secrets/kubernetes.io/serviceaccount/token';
const SA_NAMESPACE_PATH = '/var/run/secrets/kubernetes.io/serviceaccount/namespace';
const CACHE_TTL_MS = 60_000;

/**
 * Returns a function that reads a file and caches the result for CACHE_TTL_MS.
 * Used to read ServiceAccount token/namespace files without hitting disk on
 * every HTTP request, while still picking up rotated tokens within 60 seconds.
 */
function createCachedFileReader(filePath: string): () => string | undefined {
  let cached: string | undefined;
  let cachedAt = 0;
  return () => {
    const now = Date.now();
    if (cached === undefined || now - cachedAt > CACHE_TTL_MS) {
      cachedAt = now;
      try {
        cached = fs.readFileSync(filePath, 'utf-8').trim() || undefined;
      } catch {
        cached = undefined;
      }
    }
    return cached;
  };
}

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

  /** Workspace name for MLflow servers with workspaces enabled (sets X-MLFLOW-WORKSPACE header) */
  workspace?: string;
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
        `  2. Add a host to your config file at ${configPath}`,
    );
  }

  // Create Databricks SDK Config - it handles all credential resolution
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
  const databricksConfig = new Config({
    host: options.host,
    token: options.databricksToken,
    profile,
    configFile: options.databricksConfigPath,
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
    getDatabricksToken: () => databricksToken,
  };
}

/**
 * Create authentication for self-hosted (OSS) MLflow.
 *
 * Resolution order (explicit credentials win over MLFLOW_TRACKING_AUTH):
 * 1. Basic Auth (username/password from options or MLFLOW_TRACKING_USERNAME/PASSWORD)
 * 2. Bearer Token (token from options or MLFLOW_TRACKING_TOKEN)
 * 3. Kubernetes auth (MLFLOW_TRACKING_AUTH=kubernetes or kubernetes-namespaced)
 * 4. No authentication
 */
function createOssAuth(options: AuthOptions): AuthProvider {
  const host = options.trackingUri;

  // Resolve explicit credentials from options or environment
  const username = options.trackingServerUsername || process.env.MLFLOW_TRACKING_USERNAME;
  const password = options.trackingServerPassword || process.env.MLFLOW_TRACKING_PASSWORD;
  const token = options.trackingServerToken || process.env.MLFLOW_TRACKING_TOKEN;

  // Explicit credentials take precedence over MLFLOW_TRACKING_AUTH,
  // matching Python SDK where KubernetesAuth skips if Authorization
  // header is already set by username/password or token.
  const hasExplicitCredentials = (username && password) || token;
  const trackingAuth = process.env.MLFLOW_TRACKING_AUTH;
  if (
    (trackingAuth === 'kubernetes' || trackingAuth === 'kubernetes-namespaced') &&
    !hasExplicitCredentials
  ) {
    return createKubernetesAuth(
      host,
      trackingAuth === 'kubernetes-namespaced',
      SA_TOKEN_PATH,
      SA_NAMESPACE_PATH,
      options.workspace,
    );
  }

  // Pre-compute auth header since credentials don't change
  let authHeader: string | undefined;
  if (username && password) {
    const encoded = Buffer.from(`${username}:${password}`).toString('base64');
    authHeader = `Basic ${encoded}`;
  } else if (token) {
    authHeader = `Bearer ${token}`;
  }

  const workspace = process.env.MLFLOW_WORKSPACE ?? options.workspace;

  // Headers provider for OSS MLflow
  // eslint-disable-next-line require-await, @typescript-eslint/require-await
  const headersProvider: HeadersProvider = async () => {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
    if (workspace) {
      headers['X-MLFLOW-WORKSPACE'] = workspace;
    }
    return headers;
  };

  return {
    getHost: () => host,
    getHeadersProvider: () => headersProvider,
    getDatabricksToken: () => undefined,
  };
}

/**
 * Try to load token and namespace from kubeconfig (~/.kube/config).
 * Requires @kubernetes/client-node.
 * Returns null if the package is unavailable or kubeconfig cannot be loaded.
 */
async function loadFromKubeconfig(): Promise<{ token?: string; namespace?: string } | null> {
  let k8s: typeof import('@kubernetes/client-node');
  try {
    k8s = await import('@kubernetes/client-node');
  } catch {
    console.warn(
      '[mlflow] @kubernetes/client-node is not installed. ' +
        'Kubeconfig fallback is disabled. ' +
        'Install it with: npm install @kubernetes/client-node',
    );
    return null;
  }

  try {
    const kc = new k8s.KubeConfig();
    kc.loadFromDefault();

    const currentContext = kc.getCurrentContext();
    const context = kc.getContextObject(currentContext);

    // Resolve token through the full auth pipeline (exec, OIDC, cloud providers)
    // using applyToFetchOptions which is the public API for credential resolution
    const fetchOpts = await kc.applyToFetchOptions({} as any);
    const resolvedHeaders = new Headers(fetchOpts.headers as HeadersInit);

    let token: string | undefined;
    const authHeader = resolvedHeaders.get('Authorization') || resolvedHeaders.get('authorization');
    if (authHeader && authHeader.toLowerCase().startsWith('bearer ')) {
      token = authHeader.substring(7).trim() || undefined;
    }

    return {
      token,
      namespace: context?.namespace || undefined,
    };
  } catch {
    return null;
  }
}

/**
 * Create authentication for Kubernetes environments.
 *
 * Resolution order (mirrors Python SDK):
 * 1. ServiceAccount mount: reads token/namespace from
 *    /var/run/secrets/kubernetes.io/serviceaccount/ with 60s TTL caching
 * 2. Kubeconfig fallback: reads ~/.kube/config via @kubernetes/client-node
 *    (via @kubernetes/client-node)
 *
 * When enableWorkspaces is true, also reads the namespace and sets
 * the X-MLFLOW-WORKSPACE header.
 */
export function createKubernetesAuth(
  host: string,
  enableWorkspaces: boolean,
  tokenPath: string = SA_TOKEN_PATH,
  namespacePath: string = SA_NAMESPACE_PATH,
  workspace?: string,
  kubeconfigLoader: () => Promise<{ token?: string; namespace?: string } | null> = loadFromKubeconfig,
): AuthProvider {
  const getTokenFromFile = createCachedFileReader(tokenPath);
  const getNamespaceFromFile = enableWorkspaces
    ? createCachedFileReader(namespacePath)
    : undefined;

  // Kubeconfig result cached with TTL to pick up context switches and token rotation
  let kubeconfigResult: { token?: string; namespace?: string } | null | undefined;
  let kubeconfigCachedAt = 0;
  async function getKubeconfig() {
    const now = Date.now();
    if (kubeconfigResult === undefined || now - kubeconfigCachedAt > CACHE_TTL_MS) {
      kubeconfigCachedAt = now;
      kubeconfigResult = await kubeconfigLoader();
    }
    return kubeconfigResult;
  }

  const headersProvider: HeadersProvider = async () => {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };

    // Token: try SA file first, then kubeconfig, otherwise throw error
    let token = getTokenFromFile();
    if (!token) {
      token = (await getKubeconfig())?.token;
    }
    if (!token) {
      throw new Error(
        'Could not determine Kubernetes credentials. ' +
          'Ensure you are running in a Kubernetes pod with a service account ' +
          'or have a valid kubeconfig with credentials set.',
      );
    }
    headers['Authorization'] = `Bearer ${token}`;

    // Workspace: explicit MLFLOW_WORKSPACE if provided will take precedence.
    // For kubernetes-namespaced, also auto-discover from SA file or kubeconfig.
    const explicitWorkspace = process.env.MLFLOW_WORKSPACE ?? workspace;
    if (explicitWorkspace) {
      headers['X-MLFLOW-WORKSPACE'] = explicitWorkspace;
    } else if (enableWorkspaces) {
      let namespace = getNamespaceFromFile?.();
      if (!namespace) {
        namespace = (await getKubeconfig())?.namespace;
      }
      if (!namespace) {
        throw new Error(
          'Could not determine Kubernetes namespace. ' +
            'Ensure you are running in a Kubernetes pod with a service account ' +
            'or have a valid kubeconfig with a namespace set in the active context.',
        );
      }
      headers['X-MLFLOW-WORKSPACE'] = namespace;
    }

    return headers;
  };

  return {
    getHost: () => host,
    getHeadersProvider: () => headersProvider,
    getDatabricksToken: () => undefined,
  };
}
