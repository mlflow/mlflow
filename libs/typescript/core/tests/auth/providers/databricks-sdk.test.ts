import { DatabricksSdkAuthProvider } from '../../../src/auth/providers/databricks-sdk';
import { Config } from '@databricks/sdk-experimental';

// Mock the Databricks SDK
jest.mock('@databricks/sdk-experimental', () => {
  return {
    Config: jest.fn(),
    PatCredentials: jest.fn(),
    M2mCredentials: jest.fn()
  };
});

const MockConfig = Config as jest.MockedClass<typeof Config>;

describe('DatabricksSdkAuthProvider', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should throw error when host is missing', () => {
      expect(() => new DatabricksSdkAuthProvider({ host: '' })).toThrow(
        'Databricks host is required'
      );
    });

    it('should throw error when neither token nor OAuth credentials are provided', () => {
      expect(
        () =>
          new DatabricksSdkAuthProvider({
            host: 'https://my-workspace.databricks.com'
          })
      ).toThrow(
        'Either token (for PAT auth) or clientId and clientSecret (for OAuth) must be provided'
      );
    });

    it('should throw error when only clientId is provided without clientSecret', () => {
      expect(
        () =>
          new DatabricksSdkAuthProvider({
            host: 'https://my-workspace.databricks.com',
            clientId: 'client-id'
          })
      ).toThrow(
        'Either token (for PAT auth) or clientId and clientSecret (for OAuth) must be provided'
      );
    });

    it('should throw error when only clientSecret is provided without clientId', () => {
      expect(
        () =>
          new DatabricksSdkAuthProvider({
            host: 'https://my-workspace.databricks.com',
            clientSecret: 'client-secret'
          })
      ).toThrow(
        'Either token (for PAT auth) or clientId and clientSecret (for OAuth) must be provided'
      );
    });

    it('should create provider with PAT credentials', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      expect(provider).toBeDefined();
      expect(MockConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          host: 'https://my-workspace.databricks.com',
          token: 'dapi123',
          loaders: []
        })
      );
    });

    it('should create provider with OAuth credentials', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        clientId: 'client-id',
        clientSecret: 'client-secret'
      });

      expect(provider).toBeDefined();
      expect(MockConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          host: 'https://my-workspace.databricks.com',
          clientId: 'client-id',
          clientSecret: 'client-secret',
          loaders: []
        })
      );
    });

    it('should prefer PAT over OAuth when both are provided', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123',
        clientId: 'client-id',
        clientSecret: 'client-secret'
      });

      expect(provider).toBeDefined();
      // PatCredentials should be used when token is present
      expect(MockConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          token: 'dapi123'
        })
      );
    });
  });

  describe('authenticate', () => {
    it('should return authorization header from SDK', async () => {
      const mockAuthenticate = jest.fn().mockImplementation((headers: Headers) => {
        headers.set('Authorization', 'Bearer mock-token');
        return Promise.resolve();
      });

      MockConfig.mockImplementation(
        () =>
          ({
            authenticate: mockAuthenticate
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      const result = await provider.authenticate();

      expect(result.authorizationHeader).toBe('Bearer mock-token');
      expect(mockAuthenticate).toHaveBeenCalledTimes(1);
    });

    it('should return empty header when SDK sets no authorization', async () => {
      const mockAuthenticate = jest.fn().mockResolvedValue(undefined);

      MockConfig.mockImplementation(
        () =>
          ({
            authenticate: mockAuthenticate
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      const result = await provider.authenticate();

      expect(result.authorizationHeader).toBe('');
    });

    it('should propagate errors from SDK authenticate', async () => {
      const mockAuthenticate = jest
        .fn()
        .mockRejectedValue(new Error('SDK authentication failed'));

      MockConfig.mockImplementation(
        () =>
          ({
            authenticate: mockAuthenticate
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      await expect(provider.authenticate()).rejects.toThrow('SDK authentication failed');
    });

    it('should call authenticate on each call (for token refresh)', async () => {
      let callCount = 0;
      const mockAuthenticate = jest.fn().mockImplementation((headers: Headers) => {
        callCount++;
        headers.set('Authorization', `Bearer token-${callCount}`);
        return Promise.resolve();
      });

      MockConfig.mockImplementation(
        () =>
          ({
            authenticate: mockAuthenticate
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      const result1 = await provider.authenticate();
      const result2 = await provider.authenticate();

      expect(result1.authorizationHeader).toBe('Bearer token-1');
      expect(result2.authorizationHeader).toBe('Bearer token-2');
      expect(mockAuthenticate).toHaveBeenCalledTimes(2);
    });
  });
});
