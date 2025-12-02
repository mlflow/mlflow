import { PersonalAccessTokenProvider } from '../../../src/auth/providers/pat';

describe('PersonalAccessTokenProvider', () => {
  it('should return Bearer token header', async () => {
    const token = 'dapi1234567890abcdef';
    const provider = new PersonalAccessTokenProvider(token);
    const result = await provider.authenticate();

    expect(result.authorizationHeader).toBe(`Bearer ${token}`);
  });

  it('should throw error when token is empty', () => {
    expect(() => new PersonalAccessTokenProvider('')).toThrow('Personal access token is required');
  });

  it('should throw error when token is undefined', () => {
    expect(() => new PersonalAccessTokenProvider(undefined as unknown as string)).toThrow(
      'Personal access token is required'
    );
  });

  it('should return consistent results on multiple calls', async () => {
    const token = 'test-token-12345';
    const provider = new PersonalAccessTokenProvider(token);

    const result1 = await provider.authenticate();
    const result2 = await provider.authenticate();

    expect(result1.authorizationHeader).toBe(`Bearer ${token}`);
    expect(result2.authorizationHeader).toBe(`Bearer ${token}`);
  });
});
