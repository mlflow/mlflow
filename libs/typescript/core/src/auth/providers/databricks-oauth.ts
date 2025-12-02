import { AuthProvider, AuthResult, DatabricksOAuthConfig, OAuthTokenResponse, CachedToken } from '../types';

/**
 * Error class for OAuth-specific errors (authentication/authorization failures).
 * These errors should not be retried as they indicate a configuration or credential issue.
 */
class OAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OAuthError';
  }
}

/**
 * Buffer time before token expiry to trigger refresh (5 minutes in milliseconds).
 */
const TOKEN_REFRESH_BUFFER_MS = 5 * 60 * 1000;

/**
 * Maximum number of refresh retries to prevent infinite loops.
 */
const MAX_REFRESH_RETRIES = 3;

/**
 * Authentication provider using Databricks OAuth client credentials flow.
 * Handles automatic token refresh before expiry.
 */
export class DatabricksOAuthProvider implements AuthProvider {
  private readonly config: Required<DatabricksOAuthConfig>;
  private cachedToken: CachedToken | null = null;
  private refreshPromise: Promise<CachedToken> | null = null;

  constructor(config: DatabricksOAuthConfig) {
    if (!config.host) {
      throw new Error('Databricks host is required for OAuth');
    }
    if (!config.clientId) {
      throw new Error('OAuth client ID is required');
    }
    if (!config.clientSecret) {
      throw new Error('OAuth client secret is required');
    }

    this.config = {
      host: config.host.replace(/\/$/, ''), // Remove trailing slash
      clientId: config.clientId,
      clientSecret: config.clientSecret,
      scopes: config.scopes ?? ['all-apis']
    };
  }

  async authenticate(): Promise<AuthResult> {
    const token = await this.getValidToken();
    return {
      authorizationHeader: `${token.tokenType} ${token.accessToken}`
    };
  }

  /**
   * Get a valid token, refreshing if necessary.
   * Handles concurrent refresh requests by deduplicating them.
   */
  private async getValidToken(): Promise<CachedToken> {
    // Check if we have a valid cached token
    if (this.cachedToken && !this.isTokenExpiringSoon(this.cachedToken)) {
      return this.cachedToken;
    }

    // Deduplicate concurrent refresh requests
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    // Trigger refresh
    this.refreshPromise = this.refreshToken();

    try {
      const token = await this.refreshPromise;
      return token;
    } finally {
      this.refreshPromise = null;
    }
  }

  /**
   * Check if token is expiring within the buffer period (5 minutes).
   */
  private isTokenExpiringSoon(token: CachedToken): boolean {
    const now = Date.now();
    return token.expiresAt <= now + TOKEN_REFRESH_BUFFER_MS;
  }

  /**
   * Refresh the OAuth token using client credentials flow.
   * Includes retry logic with exponential backoff.
   */
  private async refreshToken(retryCount: number = 0): Promise<CachedToken> {
    if (retryCount >= MAX_REFRESH_RETRIES) {
      throw new Error(`Failed to refresh OAuth token after ${MAX_REFRESH_RETRIES} attempts`);
    }

    const tokenEndpoint = this.getTokenEndpoint();
    const body = new URLSearchParams({
      grant_type: 'client_credentials',
      scope: this.config.scopes.join(' ')
    });

    // Create Basic auth header for client credentials
    const basicAuth = Buffer.from(`${this.config.clientId}:${this.config.clientSecret}`).toString(
      'base64'
    );

    try {
      const response = await fetch(tokenEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Authorization: `Basic ${basicAuth}`
        },
        body: body.toString()
      });

      if (!response.ok) {
        const errorText = await response.text();
        // Don't retry on OAuth errors (4xx/5xx responses) - these are not transient
        throw new OAuthError(
          `OAuth token request failed: ${response.status} ${response.statusText} - ${errorText}`
        );
      }

      const tokenResponse: OAuthTokenResponse = await response.json();

      // Calculate expiry timestamp
      const expiresAt = Date.now() + tokenResponse.expires_in * 1000;

      this.cachedToken = {
        accessToken: tokenResponse.access_token,
        tokenType: tokenResponse.token_type || 'Bearer',
        expiresAt
      };

      return this.cachedToken;
    } catch (error) {
      // Don't retry on OAuth errors (authentication/authorization failures)
      if (error instanceof OAuthError) {
        throw error;
      }

      // Retry on transient network errors
      if (retryCount < MAX_REFRESH_RETRIES - 1) {
        // Exponential backoff: 100ms, 200ms, 400ms
        await this.sleep(100 * Math.pow(2, retryCount));
        return this.refreshToken(retryCount + 1);
      }
      throw error;
    }
  }

  /**
   * Get the OAuth token endpoint for the Databricks workspace.
   */
  private getTokenEndpoint(): string {
    return `${this.config.host}/oidc/v1/token`;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
