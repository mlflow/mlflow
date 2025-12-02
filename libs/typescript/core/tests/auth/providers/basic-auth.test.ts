import { BasicAuthProvider } from '../../../src/auth/providers/basic-auth';

describe('BasicAuthProvider', () => {
  it('should return Basic auth header with base64 encoded credentials', async () => {
    const username = 'testuser';
    const password = 'testpass';
    const provider = new BasicAuthProvider(username, password);
    const result = await provider.authenticate();

    const expectedCredentials = Buffer.from(`${username}:${password}`).toString('base64');
    expect(result.authorizationHeader).toBe(`Basic ${expectedCredentials}`);
  });

  it.each([
    ['empty username', '', 'password', 'Username is required for basic authentication'],
    ['undefined username', undefined, 'password', 'Username is required for basic authentication'],
    ['empty password', 'username', '', 'Password is required for basic authentication'],
    ['undefined password', 'username', undefined, 'Password is required for basic authentication']
  ])('should throw error when %s', (_, username, password, expectedError) => {
    expect(() => new BasicAuthProvider(username as string, password as string)).toThrow(
      expectedError
    );
  });

  it('should handle special characters in credentials', async () => {
    const username = 'user@domain.com';
    const password = 'p@ss:word!123';
    const provider = new BasicAuthProvider(username, password);
    const result = await provider.authenticate();

    const expectedCredentials = Buffer.from(`${username}:${password}`).toString('base64');
    expect(result.authorizationHeader).toBe(`Basic ${expectedCredentials}`);
  });
});
