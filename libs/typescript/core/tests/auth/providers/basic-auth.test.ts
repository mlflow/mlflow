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

  it('should throw error when username is empty', () => {
    expect(() => new BasicAuthProvider('', 'password')).toThrow(
      'Username is required for basic authentication'
    );
  });

  it('should throw error when password is empty', () => {
    expect(() => new BasicAuthProvider('username', '')).toThrow(
      'Password is required for basic authentication'
    );
  });

  it('should throw error when username is undefined', () => {
    expect(() => new BasicAuthProvider(undefined as unknown as string, 'password')).toThrow(
      'Username is required for basic authentication'
    );
  });

  it('should throw error when password is undefined', () => {
    expect(() => new BasicAuthProvider('username', undefined as unknown as string)).toThrow(
      'Password is required for basic authentication'
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

  it('should return consistent results on multiple calls', async () => {
    const provider = new BasicAuthProvider('user', 'pass');

    const result1 = await provider.authenticate();
    const result2 = await provider.authenticate();

    expect(result1.authorizationHeader).toBe(result2.authorizationHeader);
  });
});
