import { DatabricksSdkAuthProvider } from '../../../src/auth/providers/databricks-sdk';
import { Config } from '@databricks/sdk-experimental';

// Mock the Databricks SDK
jest.mock('@databricks/sdk-experimental', () => {
  return {
    Config: jest.fn()
  };
});

const MockConfig = Config as jest.MockedClass<typeof Config>;

describe('DatabricksSdkAuthProvider', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create provider with no options (auto-discovery)', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider();

      expect(provider).toBeDefined();
      expect(MockConfig).toHaveBeenCalledWith({
        profile: undefined,
        configFile: undefined,
        host: undefined,
        token: undefined,
        clientId: undefined,
        clientSecret: undefined
      });
    });

    it('should pass profile option to SDK Config', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider({
        profile: 'my-profile'
      });

      expect(provider).toBeDefined();
      expect(MockConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          profile: 'my-profile'
        })
      );
    });

    it('should pass configFile option to SDK Config', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider({
        configFile: '/custom/path/.databrickscfg'
      });

      expect(provider).toBeDefined();
      expect(MockConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          configFile: '/custom/path/.databrickscfg'
        })
      );
    });

    it('should pass explicit host and token to SDK Config', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      expect(provider).toBeDefined();
      expect(MockConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          host: 'https://my-workspace.databricks.com',
          token: 'dapi123'
        })
      );
    });

    it('should pass OAuth credentials to SDK Config', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      const provider = new DatabricksSdkAuthProvider({
        clientId: 'client-id',
        clientSecret: 'client-secret'
      });

      expect(provider).toBeDefined();
      expect(MockConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          clientId: 'client-id',
          clientSecret: 'client-secret'
        })
      );
    });

    it('should not set loaders (enables auto-discovery)', () => {
      MockConfig.mockImplementation(() => ({}) as any);

      new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      // Verify loaders is NOT passed (undefined, not empty array)
      const configCall = MockConfig.mock.calls[0][0];
      expect(configCall).not.toHaveProperty('loaders');
    });
  });

  describe('authenticate', () => {
    it('should call ensureResolved before authenticate', async () => {
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockAuthenticate = jest.fn().mockImplementation((headers: Headers) => {
        headers.set('Authorization', 'Bearer mock-token');
        return Promise.resolve();
      });

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
            authenticate: mockAuthenticate
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      await provider.authenticate();

      expect(mockEnsureResolved).toHaveBeenCalled();
      expect(mockAuthenticate).toHaveBeenCalled();
    });

    it('should return authorization header from SDK', async () => {
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockAuthenticate = jest.fn().mockImplementation((headers: Headers) => {
        headers.set('Authorization', 'Bearer mock-token');
        return Promise.resolve();
      });

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
            authenticate: mockAuthenticate
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider({
        host: 'https://my-workspace.databricks.com',
        token: 'dapi123'
      });

      const result = await provider.authenticate();

      expect(result.authorizationHeader).toBe('Bearer mock-token');
    });

    it('should return empty header when SDK sets no authorization', async () => {
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockAuthenticate = jest.fn().mockResolvedValue(undefined);

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
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
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockAuthenticate = jest
        .fn()
        .mockRejectedValue(new Error('SDK authentication failed'));

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
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
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockAuthenticate = jest.fn().mockImplementation((headers: Headers) => {
        callCount++;
        headers.set('Authorization', `Bearer token-${callCount}`);
        return Promise.resolve();
      });

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
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

  describe('getHost', () => {
    it('should call ensureResolved before getHost', async () => {
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockGetHost = jest.fn().mockResolvedValue(new URL('https://my-workspace.databricks.com'));

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
            getHost: mockGetHost
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider();

      await provider.getHost();

      expect(mockEnsureResolved).toHaveBeenCalled();
      expect(mockGetHost).toHaveBeenCalled();
    });

    it('should return host URL as string', async () => {
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockGetHost = jest.fn().mockResolvedValue(new URL('https://my-workspace.databricks.com'));

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
            getHost: mockGetHost
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider();

      const host = await provider.getHost();

      expect(host).toBe('https://my-workspace.databricks.com/');
    });

    it('should propagate errors from SDK getHost', async () => {
      const mockEnsureResolved = jest.fn().mockResolvedValue(undefined);
      const mockGetHost = jest.fn().mockRejectedValue(new Error('host is not set'));

      MockConfig.mockImplementation(
        () =>
          ({
            ensureResolved: mockEnsureResolved,
            getHost: mockGetHost
          }) as any
      );

      const provider = new DatabricksSdkAuthProvider();

      await expect(provider.getHost()).rejects.toThrow('host is not set');
    });
  });
});
