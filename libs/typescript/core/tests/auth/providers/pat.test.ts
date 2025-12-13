import { PersonalAccessTokenProvider } from '../../../src/auth/providers/pat';

describe('PersonalAccessTokenProvider', () => {
  it('should return Bearer token header', async () => {
    const token = 'dapi1234567890abcdef';
    const provider = new PersonalAccessTokenProvider(token);
    const result = await provider.authenticate();

    expect(result.authorizationHeader).toBe(`Bearer ${token}`);
  });

  it.each([
    ['empty', ''],
    ['undefined', undefined]
  ])('should throw error when token is %s', (_, token) => {
    expect(() => new PersonalAccessTokenProvider(token as string)).toThrow(
      'Personal access token is required'
    );
  });
});
