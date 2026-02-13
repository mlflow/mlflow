import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { fetchAPI } from './FetchUtils';

describe('fetchAPI error handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('response body in error messages', () => {
    it('should include response body in error message when request fails', async () => {
      const errorResponseBody = JSON.stringify({
        error_code: 'RESOURCE_DOES_NOT_EXIST',
        message: 'Run with ID abc123 not found',
      });

      const mockResponse = {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        text: jest.fn(() => Promise.resolve(errorResponseBody)),
      };

      const mockFetch = jest.fn(() => Promise.resolve(mockResponse));
      global.fetch = mockFetch as any;

      await expect(fetchAPI('/api/2.0/mlflow/runs/get')).rejects.toThrow(`HTTP 404: Not Found - ${errorResponseBody}`);

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should include response body with JSON error details', async () => {
      const errorResponseBody = JSON.stringify({
        error_code: 'INVALID_PARAMETER_VALUE',
        message: 'Invalid experiment ID: must be a positive integer',
      });

      const mockResponse = {
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        text: jest.fn(() => Promise.resolve(errorResponseBody)),
      };

      const mockFetch = jest.fn(() => Promise.resolve(mockResponse));
      global.fetch = mockFetch as any;

      await expect(
        fetchAPI('/api/2.0/mlflow/experiments/get', {
          method: 'GET',
        }),
      ).rejects.toThrow(`HTTP 400: Bad Request - ${errorResponseBody}`);

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should handle empty response body in error', async () => {
      const mockResponse = {
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        text: jest.fn(() => Promise.resolve('')),
      };

      const mockFetch = jest.fn(() => Promise.resolve(mockResponse));
      global.fetch = mockFetch as any;

      await expect(fetchAPI('/api/2.0/mlflow/runs/list')).rejects.toThrow('HTTP 401: Unauthorized');

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should handle error when reading response body fails', async () => {
      const mockResponse = {
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: jest.fn(() => Promise.reject(new Error('Failed to read body'))),
      };

      const mockFetch = jest.fn(() => Promise.resolve(mockResponse));
      global.fetch = mockFetch as any;

      await expect(
        fetchAPI('/api/2.0/mlflow/runs/create', {
          method: 'POST',
          body: { experiment_id: '1' },
        }),
      ).rejects.toThrow('HTTP 500: Internal Server Error');

      expect(mockResponse.text).toHaveBeenCalled();
    });

    it('should include plain text response body in error', async () => {
      const errorResponseBody = 'Gateway timeout - upstream service not responding';

      const mockResponse = {
        ok: false,
        status: 504,
        statusText: 'Gateway Timeout',
        text: jest.fn(() => Promise.resolve(errorResponseBody)),
      };

      const mockFetch = jest.fn(() => Promise.resolve(mockResponse));
      global.fetch = mockFetch as any;

      await expect(fetchAPI('/api/2.0/mlflow/traces/search')).rejects.toThrow(
        `HTTP 504: Gateway Timeout - ${errorResponseBody}`,
      );

      expect(mockResponse.text).toHaveBeenCalled();
    });
  });
});
