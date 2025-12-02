import { NoAuthProvider } from '../../../src/auth/providers/no-auth';

describe('NoAuthProvider', () => {
  it('should return empty authorization header', async () => {
    const provider = new NoAuthProvider();
    const result = await provider.authenticate();

    expect(result.authorizationHeader).toBe('');
  });

  it('should return consistent results on multiple calls', async () => {
    const provider = new NoAuthProvider();

    const result1 = await provider.authenticate();
    const result2 = await provider.authenticate();

    expect(result1.authorizationHeader).toBe('');
    expect(result2.authorizationHeader).toBe('');
  });
});
