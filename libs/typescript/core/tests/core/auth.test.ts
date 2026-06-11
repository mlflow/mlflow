import { createAuthProvider } from '../../src/auth/index';

describe('createOssAuth', () => {
  afterEach(() => {
    delete process.env.MLFLOW_TRACKING_TOKEN;
    delete process.env.MLFLOW_TRACKING_USERNAME;
    delete process.env.MLFLOW_TRACKING_PASSWORD;
    delete process.env.MLFLOW_WORKSPACE;
  });

  it('should include x-mlflow-workspace header when MLFLOW_WORKSPACE is set', async () => {
    process.env.MLFLOW_WORKSPACE = 'my-namespace';

    const provider = createAuthProvider({ trackingUri: 'http://mlflow.example.com:5000' });
    const headers = await provider.getHeadersProvider()();

    expect(headers['x-mlflow-workspace']).toBe('my-namespace');
  });

  it('should not include x-mlflow-workspace header when MLFLOW_WORKSPACE is not set', async () => {
    const provider = createAuthProvider({ trackingUri: 'http://mlflow.example.com:5000' });
    const headers = await provider.getHeadersProvider()();

    expect(headers['x-mlflow-workspace']).toBeUndefined();
  });

  it('should include both Authorization and x-mlflow-workspace headers together', async () => {
    process.env.MLFLOW_TRACKING_TOKEN = 'my-token';
    process.env.MLFLOW_WORKSPACE = 'my-namespace';

    const provider = createAuthProvider({ trackingUri: 'http://mlflow.example.com:5000' });
    const headers = await provider.getHeadersProvider()();

    expect(headers['Authorization']).toBe('Bearer my-token');
    expect(headers['x-mlflow-workspace']).toBe('my-namespace');
  });

  it('should always include Content-Type header', async () => {
    const provider = createAuthProvider({ trackingUri: 'http://mlflow.example.com:5000' });
    const headers = await provider.getHeadersProvider()();

    expect(headers['Content-Type']).toBe('application/json');
  });
});
