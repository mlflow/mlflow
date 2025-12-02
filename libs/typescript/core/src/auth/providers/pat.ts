import { AuthProvider, AuthResult } from '../types';

/**
 * Authentication provider using Databricks Personal Access Token (PAT).
 * Provides static Bearer token authentication.
 */
export class PersonalAccessTokenProvider implements AuthProvider {
  private readonly token: string;

  constructor(token: string) {
    if (!token) {
      throw new Error('Personal access token is required');
    }
    this.token = token;
  }

  authenticate(): Promise<AuthResult> {
    return Promise.resolve({
      authorizationHeader: `Bearer ${this.token}`
    });
  }
}
