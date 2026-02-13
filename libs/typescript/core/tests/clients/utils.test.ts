import { makeRequest } from '../../src/clients/utils';

describe('makeRequest', () => {
  const mockHeaderProvider = () => Promise.resolve({ 'Content-Type': 'application/json' });

  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('error handling with response body', () => {
    it('should include response body in error message when request fails', async () => {
      const errorResponseBody = JSON.stringify({
        error_code: 'INVALID_PARAMETER_VALUE',
        message: 'Invalid experiment ID provided',
      });

      const mockResponse = {
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        text: jest.fn().mockResolvedValue(errorResponseBody),
      };

      jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse as any);

      await expect(
        makeRequest(
          'GET',
          'http://localhost:5000/api/2.0/mlflow/experiments/get',
          mockHeaderProvider,
        ),
      ).rejects.toThrow(`HTTP 400: Bad Request - ${errorResponseBody}`);

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should handle empty response body in error', async () => {
      const mockResponse = {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        text: jest.fn().mockResolvedValue(''),
      };

      jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse as any);

      await expect(
        makeRequest(
          'GET',
          'http://localhost:5000/api/2.0/mlflow/experiments/get',
          mockHeaderProvider,
        ),
      ).rejects.toThrow('HTTP 404: Not Found');

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should handle error when reading response body fails', async () => {
      const mockResponse = {
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: jest.fn().mockRejectedValue(new Error('Failed to read body')),
      };

      jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse as any);

      await expect(
        makeRequest(
          'GET',
          'http://localhost:5000/api/2.0/mlflow/experiments/get',
          mockHeaderProvider,
        ),
      ).rejects.toThrow('HTTP 500: Internal Server Error');

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should include response body with plain text error', async () => {
      const errorResponseBody = 'Service temporarily unavailable';

      const mockResponse = {
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: jest.fn().mockResolvedValue(errorResponseBody),
      };

      jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse as any);

      await expect(
        makeRequest(
          'POST',
          'http://localhost:5000/api/2.0/mlflow/runs/create',
          mockHeaderProvider,
          {},
        ),
      ).rejects.toThrow(`HTTP 503: Service Unavailable - ${errorResponseBody}`);

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should truncate large response bodies to prevent memory issues', async () => {
      // Create a response body larger than 10KB
      const largeResponseBody = 'x'.repeat(15 * 1024); // 15KB

      const mockResponse = {
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: jest.fn().mockResolvedValue(largeResponseBody),
      };

      jest.spyOn(global, 'fetch').mockResolvedValue(mockResponse as any);

      await expect(
        makeRequest(
          'POST',
          'http://localhost:5000/api/2.0/mlflow/runs/create',
          mockHeaderProvider,
          {},
        ),
      ).rejects.toThrow(/HTTP 500: Internal Server Error - x{10240}\.\.\. \(truncated\)/);

      expect(mockResponse.text).toHaveBeenCalled();
    });
  });
});
