import { AuthProvider, AuthResult } from '../types';

/**
 * Authentication provider using HTTP Basic Authentication.
 * Used for MLflow tracking server with username/password authentication.
 */
export class BasicAuthProvider implements AuthProvider {
  private readonly username: string;
  private readonly password: string;

  constructor(username: string, password: string) {
    if (!username) {
      throw new Error('Username is required for basic authentication');
    }
    if (!password) {
      throw new Error('Password is required for basic authentication');
    }
    this.username = username;
    this.password = password;
  }

  authenticate(): Promise<AuthResult> {
    const credentials = Buffer.from(`${this.username}:${this.password}`).toString('base64');
    return Promise.resolve({
      authorizationHeader: `Basic ${credentials}`
    });
  }
}
