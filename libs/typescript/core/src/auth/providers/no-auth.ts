import { AuthProvider, AuthResult } from '../types';

/**
 * No-op authentication provider for backwards compatibility.
 * Used when no authentication is required (e.g., local MLflow server without auth).
 */
export class NoAuthProvider implements AuthProvider {
  authenticate(): Promise<AuthResult> {
    return Promise.resolve({
      authorizationHeader: ''
    });
  }
}
