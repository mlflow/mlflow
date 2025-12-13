import { NoAuthProvider } from '../../../src/auth/providers/no-auth';

describe('NoAuthProvider', () => {
  it('should return empty authorization header', async () => {
    const provider = new NoAuthProvider();
    const result = await provider.authenticate();

    expect(result.authorizationHeader).toBe('');
  });
});
