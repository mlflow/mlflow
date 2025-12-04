import { Config, PatCredentials } from '@databricks/sdk-experimental';
import type { AuthType } from '@databricks/sdk-experimental';
import { AuthProvider, AuthResult } from '../types';

/**
 * Configuration options for DatabricksSdkAuthProvider.
 */
export interface DatabricksSdkAuthConfig {
  /**
   * The Databricks workspace host URL (e.g., "https://my-workspace.databricks.com")
   */
  host: string;

  /**
   * Databricks Personal Access Token (PAT) for authentication.
   * If provided, PAT authentication will be used.
   */
  token?: string;

  /**
   * OAuth client ID for service principal authentication.
   * Must be provided together with clientSecret for OAuth M2M flow.
   */
  clientId?: string;

  /**
   * OAuth client secret for service principal authentication.
   * Must be provided together with clientId for OAuth M2M flow.
   */
  clientSecret?: string;
}

/**
 * Authentication provider that uses the official Databricks SDK for authentication.
 *
 * This provider wraps the @databricks/sdk-experimental package's Config class
 * to implement MLflow's AuthProvider interface. It supports:
 * - Personal Access Token (PAT) authentication
 * - OAuth M2M (client credentials) authentication with automatic token refresh
 *
 * The SDK handles token caching and refresh automatically with a 40-second
 * buffer before expiry.
 *
 * @example PAT Authentication
 * ```typescript
 * const provider = new DatabricksSdkAuthProvider({
 *   host: 'https://my-workspace.databricks.com',
 *   token: process.env.DATABRICKS_TOKEN
 * });
 * ```
 *
 * @example OAuth M2M Authentication
 * ```typescript
 * const provider = new DatabricksSdkAuthProvider({
 *   host: 'https://my-workspace.databricks.com',
 *   clientId: process.env.DATABRICKS_CLIENT_ID,
 *   clientSecret: process.env.DATABRICKS_CLIENT_SECRET
 * });
 * ```
 */
export class DatabricksSdkAuthProvider implements AuthProvider {
  private config: Config;

  constructor(options: DatabricksSdkAuthConfig) {
    if (!options.host) {
      throw new Error('Databricks host is required');
    }

    // Validate that at least one auth method is provided
    const hasPat = !!options.token;
    const hasOAuth = !!(options.clientId && options.clientSecret);

    if (!hasPat && !hasOAuth) {
      throw new Error(
        'Either token (for PAT auth) or clientId and clientSecret (for OAuth) must be provided'
      );
    }

    // Determine auth type and credentials based on provided options
    // PatCredentials is exported, but M2mCredentials is not - so for OAuth we use
    // the DefaultCredentials chain with authType set to force the right provider
    const authType: AuthType = hasPat ? 'pat' : 'oauth-m2m';
    const credentials = hasPat ? new PatCredentials() : undefined;

    this.config = new Config({
      host: options.host,
      token: options.token,
      clientId: options.clientId,
      clientSecret: options.clientSecret,
      authType, // Force specific auth type to avoid auto-detect
      credentials, // Use explicit credentials for PAT, default chain for OAuth
      loaders: [] // Disable env/config file loaders for explicit-only auth
    });
  }

  /**
   * Authenticate and return the authorization header.
   *
   * For PAT auth, this returns immediately with a Bearer token.
   * For OAuth M2M, this handles token exchange and automatic refresh
   * (tokens are refreshed 40 seconds before expiry).
   *
   * @returns Promise resolving to AuthResult with the Authorization header value
   * @throws Error if authentication fails
   */
  async authenticate(): Promise<AuthResult> {
    // SDK's authenticate() uses refreshableTokenVisitor internally
    // which handles 40-second refresh buffer automatically
    const headers = new Headers();
    await this.config.authenticate(headers);

    return {
      authorizationHeader: headers.get('Authorization') || ''
    };
  }
}
