import { Config } from '@databricks/sdk-experimental';
import { AuthProvider, AuthResult } from '../types';

/**
 * Configuration options for DatabricksSdkAuthProvider.
 *
 * All options are optional - the SDK will auto-discover credentials from:
 * 1. Environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN, etc.)
 * 2. Config file (~/.databrickscfg)
 * 3. Cloud provider auth (Azure CLI, GCP, etc.)
 *
 * Explicit options override auto-discovered values.
 */
export interface DatabricksSdkAuthConfig {
  /**
   * Profile name to use from the Databricks config file.
   * Extracted from tracking URI like "databricks://my-profile".
   * If not specified, uses the DEFAULT profile.
   */
  profile?: string;

  /**
   * Custom path to the Databricks config file.
   * Defaults to ~/.databrickscfg if not specified.
   */
  configFile?: string;

  /**
   * Explicit Databricks workspace host URL.
   * If provided, overrides auto-discovered host.
   */
  host?: string;

  /**
   * Explicit Personal Access Token (PAT).
   * If provided, uses PAT authentication.
   */
  token?: string;

  /**
   * OAuth client ID for service principal authentication.
   * Used with clientSecret for OAuth M2M flow (e.g., Databricks Apps).
   */
  clientId?: string;

  /**
   * OAuth client secret for service principal authentication.
   * Used with clientId for OAuth M2M flow.
   */
  clientSecret?: string;
}

/**
 * Authentication provider that delegates to the official Databricks SDK.
 *
 * This provider wraps the @databricks/sdk-experimental package's Config class
 * to implement MLflow's AuthProvider interface. It supports all authentication
 * methods supported by the Databricks SDK:
 *
 * - Personal Access Token (PAT) authentication
 * - OAuth M2M (client credentials) for service principals
 * - Azure CLI authentication
 * - Azure Managed Service Identity (MSI)
 * - Azure Service Principal (client secret)
 * - GCP default credentials
 * - GCP service account impersonation
 * - Databricks CLI credentials
 * - Metadata service credentials
 *
 * The SDK handles credential discovery automatically by trying each method
 * in order until one succeeds. It also handles OAuth token refresh automatically
 * with a 40-second buffer before expiry.
 *
 * @example Automatic discovery (recommended)
 * ```typescript
 * // SDK auto-discovers from env vars or ~/.databrickscfg
 * const provider = new DatabricksSdkAuthProvider();
 * ```
 *
 * @example Named profile
 * ```typescript
 * // Use a specific profile from ~/.databrickscfg
 * const provider = new DatabricksSdkAuthProvider({
 *   profile: 'my-profile'
 * });
 * ```
 *
 * @example Explicit credentials (for testing or override)
 * ```typescript
 * const provider = new DatabricksSdkAuthProvider({
 *   host: 'https://my-workspace.databricks.com',
 *   token: 'dapi...'
 * });
 * ```
 *
 * @example OAuth M2M (Databricks Apps)
 * ```typescript
 * // Environment typically provides DATABRICKS_CLIENT_ID/SECRET
 * const provider = new DatabricksSdkAuthProvider({
 *   clientId: process.env.DATABRICKS_CLIENT_ID,
 *   clientSecret: process.env.DATABRICKS_CLIENT_SECRET
 * });
 * ```
 */
export class DatabricksSdkAuthProvider implements AuthProvider {
  private config: Config;

  constructor(options: DatabricksSdkAuthConfig = {}) {
    this.config = new Config({
      profile: options.profile,
      configFile: options.configFile,
      host: options.host,
      token: options.token,
      clientId: options.clientId,
      clientSecret: options.clientSecret
      // Do NOT set loaders: [] - let SDK use default loaders for auto-discovery
    });
  }

  /**
   * Authenticate and return the authorization header.
   *
   * The SDK handles all authentication methods and token refresh automatically.
   * For OAuth flows, tokens are refreshed 40 seconds before expiry.
   *
   * @returns Promise resolving to AuthResult with the Authorization header value
   * @throws Error if authentication fails or no credentials can be discovered
   */
  async authenticate(): Promise<AuthResult> {
    await this.config.ensureResolved();
    const headers = new Headers();
    await this.config.authenticate(headers);

    return {
      authorizationHeader: headers.get('Authorization') || ''
    };
  }

  /**
   * Get the resolved Databricks workspace host URL.
   *
   * This resolves the host from:
   * 1. Explicit host option (if provided)
   * 2. Environment variables (DATABRICKS_HOST)
   * 3. Config file profile
   *
   * @returns Promise resolving to the workspace host URL (e.g., "https://my-workspace.databricks.com")
   * @throws Error if host cannot be resolved
   */
  async getHost(): Promise<string> {
    await this.config.ensureResolved();
    const host = await this.config.getHost();
    return host.toString();
  }
}
