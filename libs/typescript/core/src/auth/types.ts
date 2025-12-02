/**
 * Result of an authentication operation.
 * Contains the authorization header value to be used in HTTP requests.
 */
export interface AuthResult {
  /**
   * The full Authorization header value (e.g., "Bearer <token>" or "Basic <credentials>").
   * Empty string indicates no authentication should be applied.
   */
  authorizationHeader: string;
}

/**
 * Interface for authentication providers.
 * Implementations handle different authentication mechanisms (PAT, OAuth, Basic Auth, etc.)
 */
export interface AuthProvider {
  /**
   * Get authentication credentials for a request.
   * This method should handle token refresh if needed.
   *
   * @returns Promise resolving to authentication result with header value
   * @throws Error if authentication fails
   */
  authenticate(): Promise<AuthResult>;
}

/**
 * Configuration for Databricks OAuth client credentials flow.
 */
export interface DatabricksOAuthConfig {
  /**
   * The Databricks workspace host URL (e.g., "https://my-workspace.databricks.com")
   */
  host: string;

  /**
   * OAuth client ID for the service principal
   */
  clientId: string;

  /**
   * OAuth client secret for the service principal
   */
  clientSecret: string;

  /**
   * Optional OAuth scopes. Defaults to ["all-apis"] if not provided.
   */
  scopes?: string[];
}

/**
 * OAuth token response from Databricks OIDC endpoint.
 */
export interface OAuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number; // seconds until expiry
  scope?: string;
}

/**
 * Internal structure for cached OAuth token with expiry tracking.
 */
export interface CachedToken {
  accessToken: string;
  tokenType: string;
  expiresAt: number; // Unix timestamp in milliseconds
}
