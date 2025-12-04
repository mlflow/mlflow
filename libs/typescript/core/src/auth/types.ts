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
