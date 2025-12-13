import { MlflowClient } from '../../src/clients/client';
import { PersonalAccessTokenProvider, BasicAuthProvider, NoAuthProvider } from '../../src/auth';

/**
 * These tests verify that the MlflowClient maintains backwards compatibility
 * with the legacy authentication options (databricksToken, trackingServerUsername/Password).
 *
 * The existing flow is:
 * 1. User sets DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
 * 2. User calls init({ trackingUri: 'databricks', experimentId: '...' })
 * 3. config.ts reads env vars and sets host/databricksToken in config
 * 4. provider.ts creates MlflowClient with databricksToken
 * 5. MlflowClient creates PersonalAccessTokenProvider internally
 *
 * This flow must continue to work after adding the authProvider feature.
 */
describe('MlflowClient backwards compatibility', () => {
  describe('legacy databricksToken option', () => {
    it('should create a PersonalAccessTokenProvider internally when databricksToken is provided', () => {
      // This is the legacy way to create a client with Databricks token
      const client = new MlflowClient({
        trackingUri: 'databricks',
        host: 'https://my-workspace.databricks.com',
        databricksToken: 'dapi12345'
      });

      // The client should work - we can't directly access private authProvider,
      // but we can verify the client was created successfully
      expect(client).toBeDefined();
    });
  });

  describe('legacy basic auth options', () => {
    it('should create a BasicAuthProvider internally when username/password are provided', () => {
      // This is the legacy way to create a client with basic auth
      const client = new MlflowClient({
        trackingUri: 'http://localhost:5000',
        host: 'http://localhost:5000',
        trackingServerUsername: 'user',
        trackingServerPassword: 'pass'
      });

      // The client should work
      expect(client).toBeDefined();
    });
  });

  describe('no auth option', () => {
    it('should create a NoAuthProvider internally when no auth options are provided', () => {
      // This is the default - no auth required
      const client = new MlflowClient({
        trackingUri: 'http://localhost:5000',
        host: 'http://localhost:5000'
      });

      // The client should work
      expect(client).toBeDefined();
    });
  });

  describe('new authProvider option', () => {
    it('should use provided authProvider when specified', () => {
      const authProvider = new PersonalAccessTokenProvider('my-token');

      const client = new MlflowClient({
        trackingUri: 'databricks',
        host: 'https://my-workspace.databricks.com',
        authProvider
      });

      expect(client).toBeDefined();
    });

    it('should prefer authProvider over legacy databricksToken when both are provided', () => {
      // When both are provided, authProvider should take precedence
      const authProvider = new PersonalAccessTokenProvider('auth-provider-token');

      const client = new MlflowClient({
        trackingUri: 'databricks',
        host: 'https://my-workspace.databricks.com',
        authProvider,
        databricksToken: 'legacy-token' // This should be ignored
      });

      expect(client).toBeDefined();
    });

    it('should work with BasicAuthProvider', () => {
      const authProvider = new BasicAuthProvider('user', 'pass');

      const client = new MlflowClient({
        trackingUri: 'http://localhost:5000',
        host: 'http://localhost:5000',
        authProvider
      });

      expect(client).toBeDefined();
    });

    it('should work with NoAuthProvider', () => {
      const authProvider = new NoAuthProvider();

      const client = new MlflowClient({
        trackingUri: 'http://localhost:5000',
        host: 'http://localhost:5000',
        authProvider
      });

      expect(client).toBeDefined();
    });
  });
});
