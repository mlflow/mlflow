import { DatabricksOAuthProvider } from '../../../src/auth/providers/databricks-oauth';

// Store original fetch to restore after tests
const originalFetch = global.fetch;
const mockFetch = jest.fn();

describe('DatabricksOAuthProvider', () => {
  beforeAll(() => {
    // Replace global fetch with mock
    global.fetch = mockFetch;
  });

  afterAll(() => {
    // Restore original fetch after all tests
    global.fetch = originalFetch;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  const defaultConfig = {
    host: 'https://my-workspace.databricks.com',
    clientId: 'test-client-id',
    clientSecret: 'test-client-secret'
  };

  describe('constructor', () => {
    it.each([
      [
        'host',
        { host: '', clientId: 'id', clientSecret: 'secret' },
        'Databricks host is required for OAuth'
      ],
      [
        'clientId',
        { host: 'https://example.com', clientId: '', clientSecret: 'secret' },
        'OAuth client ID is required'
      ],
      [
        'clientSecret',
        { host: 'https://example.com', clientId: 'id', clientSecret: '' },
        'OAuth client secret is required'
      ]
    ])('should throw error when %s is missing', (_, config, expectedError) => {
      expect(() => new DatabricksOAuthProvider(config)).toThrow(expectedError);
    });

    it('should strip trailing slash from host', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            access_token: 'token123',
            token_type: 'Bearer',
            expires_in: 3600
          })
      });

      const provider = new DatabricksOAuthProvider({
        ...defaultConfig,
        host: 'https://my-workspace.databricks.com/'
      });

      await provider.authenticate();

      expect(mockFetch).toHaveBeenCalledWith(
        'https://my-workspace.databricks.com/oidc/v1/token',
        expect.any(Object)
      );
    });
  });

  describe('authenticate', () => {
    it('should fetch token and return Bearer header', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            access_token: 'test-access-token',
            token_type: 'Bearer',
            expires_in: 3600
          })
      });

      const provider = new DatabricksOAuthProvider(defaultConfig);
      const result = await provider.authenticate();

      expect(result.authorizationHeader).toBe('Bearer test-access-token');
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://my-workspace.databricks.com/oidc/v1/token',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/x-www-form-urlencoded',
            Authorization: expect.stringMatching(/^Basic /)
          })
        })
      );
    });

    it('should use custom scopes when provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            access_token: 'token',
            token_type: 'Bearer',
            expires_in: 3600
          })
      });

      const provider = new DatabricksOAuthProvider({
        ...defaultConfig,
        scopes: ['scope1', 'scope2']
      });

      await provider.authenticate();

      const [, options] = mockFetch.mock.calls[0];
      expect(options.body).toContain('scope=scope1+scope2');
    });

    it('should use default scopes when not provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            access_token: 'token',
            token_type: 'Bearer',
            expires_in: 3600
          })
      });

      const provider = new DatabricksOAuthProvider(defaultConfig);
      await provider.authenticate();

      const [, options] = mockFetch.mock.calls[0];
      expect(options.body).toContain('scope=all-apis');
    });
  });

  describe('token caching', () => {
    it('should cache token and not refetch when valid', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            access_token: 'cached-token',
            token_type: 'Bearer',
            expires_in: 3600
          })
      });

      const provider = new DatabricksOAuthProvider(defaultConfig);

      const result1 = await provider.authenticate();
      const result2 = await provider.authenticate();

      expect(result1.authorizationHeader).toBe('Bearer cached-token');
      expect(result2.authorizationHeader).toBe('Bearer cached-token');
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should refresh token when expiring within 5 minutes', async () => {
      // First token expires in 4 minutes (240 seconds) - should trigger refresh
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () =>
            Promise.resolve({
              access_token: 'old-token',
              token_type: 'Bearer',
              expires_in: 240
            })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () =>
            Promise.resolve({
              access_token: 'new-token',
              token_type: 'Bearer',
              expires_in: 3600
            })
        });

      const provider = new DatabricksOAuthProvider(defaultConfig);

      // First call - gets old token
      const result1 = await provider.authenticate();
      expect(result1.authorizationHeader).toBe('Bearer old-token');

      // Advance time by 1 second - token should still be considered expiring soon
      jest.advanceTimersByTime(1000);

      // Second call - should refresh because token is expiring within 5 minutes
      const result2 = await provider.authenticate();
      expect(result2.authorizationHeader).toBe('Bearer new-token');
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('should refresh token when already expired', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () =>
            Promise.resolve({
              access_token: 'first-token',
              token_type: 'Bearer',
              expires_in: 1
            })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () =>
            Promise.resolve({
              access_token: 'refreshed-token',
              token_type: 'Bearer',
              expires_in: 3600
            })
        });

      const provider = new DatabricksOAuthProvider(defaultConfig);

      await provider.authenticate();

      // Advance time past expiry
      jest.advanceTimersByTime(2000);

      const result = await provider.authenticate();
      expect(result.authorizationHeader).toBe('Bearer refreshed-token');
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  describe('request deduplication', () => {
    it('should deduplicate concurrent refresh requests', async () => {
      let resolvePromise: (value: unknown) => void;
      const delayedPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });

      mockFetch.mockImplementationOnce(() =>
        delayedPromise.then(() => ({
          ok: true,
          json: () =>
            Promise.resolve({
              access_token: 'deduped-token',
              token_type: 'Bearer',
              expires_in: 3600
            })
        }))
      );

      const provider = new DatabricksOAuthProvider(defaultConfig);

      // Start multiple concurrent requests
      const promise1 = provider.authenticate();
      const promise2 = provider.authenticate();
      const promise3 = provider.authenticate();

      // Resolve the fetch
      resolvePromise!(undefined);

      const [result1, result2, result3] = await Promise.all([promise1, promise2, promise3]);

      expect(result1.authorizationHeader).toBe('Bearer deduped-token');
      expect(result2.authorizationHeader).toBe('Bearer deduped-token');
      expect(result3.authorizationHeader).toBe('Bearer deduped-token');
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('error handling', () => {
    it('should throw error on OAuth failure without retrying', async () => {
      // HTTP errors (non-ok responses) should not trigger retries
      mockFetch.mockResolvedValue({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        text: () => Promise.resolve('Invalid client credentials')
      });

      const provider = new DatabricksOAuthProvider(defaultConfig);

      // Run all pending promises to completion
      await expect(provider.authenticate()).rejects.toThrow(
        'OAuth token request failed: 401 Unauthorized - Invalid client credentials'
      );
    });

    it('should retry on transient errors with exponential backoff', async () => {
      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          ok: true,
          json: () =>
            Promise.resolve({
              access_token: 'success-token',
              token_type: 'Bearer',
              expires_in: 3600
            })
        });

      const provider = new DatabricksOAuthProvider(defaultConfig);

      const resultPromise = provider.authenticate();

      // First retry after 100ms
      await jest.advanceTimersByTimeAsync(100);
      // Second retry after 200ms
      await jest.advanceTimersByTimeAsync(200);

      const result = await resultPromise;
      expect(result.authorizationHeader).toBe('Bearer success-token');
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    it('should fail after max retries', async () => {
      // Use real timers for this test since we need to actually wait for retries
      jest.useRealTimers();

      try {
        const networkError = new Error('Network error');
        mockFetch.mockRejectedValue(networkError);

        const provider = new DatabricksOAuthProvider(defaultConfig);

        // Verify it throws after exhausting retries
        await expect(provider.authenticate()).rejects.toThrow('Network error');
        expect(mockFetch).toHaveBeenCalledTimes(3);
      } finally {
        // Always restore fake timers, even if test fails
        jest.useFakeTimers();
      }
    });
  });

  describe('Basic auth header', () => {
    it('should encode client credentials as Basic auth', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            access_token: 'token',
            token_type: 'Bearer',
            expires_in: 3600
          })
      });

      const provider = new DatabricksOAuthProvider(defaultConfig);
      await provider.authenticate();

      const [, options] = mockFetch.mock.calls[0];
      const expectedAuth = Buffer.from(
        `${defaultConfig.clientId}:${defaultConfig.clientSecret}`
      ).toString('base64');
      expect(options.headers.Authorization).toBe(`Basic ${expectedAuth}`);
    });
  });
});
