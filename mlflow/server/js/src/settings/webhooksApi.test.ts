import { jest, describe, it, expect, beforeEach } from '@jest/globals';

import { fetchAPI } from '../common/utils/FetchUtils';
import { WebhooksApi } from './webhooksApi';

jest.mock('../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(() => Promise.resolve()),
  getAjaxUrl: jest.fn((url: string) => url),
  HTTPMethods: { GET: 'GET', POST: 'POST', PATCH: 'PATCH', DELETE: 'DELETE' },
}));

const mockFetchAPI = jest.mocked(fetchAPI);

describe('WebhooksApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('listWebhooks', () => {
    it('calls fetchAPI with the correct URL', async () => {
      const mockResponse = { webhooks: [{ webhook_id: 'wh-1', name: 'Test' }] };
      mockFetchAPI.mockResolvedValue(mockResponse as any);

      const result = await WebhooksApi.listWebhooks();

      expect(mockFetchAPI).toHaveBeenCalledWith('ajax-api/2.0/mlflow/webhooks');
      expect(result).toEqual(mockResponse);
    });
  });

  describe('createWebhook', () => {
    it('calls fetchAPI with POST method and data', async () => {
      const request = {
        name: 'New Webhook',
        url: 'https://example.com/hook',
        events: [{ entity: 'PROMPT', action: 'CREATED' }],
        status: 'ACTIVE' as const,
      };
      mockFetchAPI.mockResolvedValue({} as any);

      await WebhooksApi.createWebhook(request);

      expect(mockFetchAPI).toHaveBeenCalledWith('ajax-api/2.0/mlflow/webhooks', {
        method: 'POST',
        body: request,
      });
    });
  });

  describe('updateWebhook', () => {
    it('calls fetchAPI with PATCH method and data', async () => {
      const request = {
        name: 'Updated Webhook',
        url: 'https://example.com/hook',
        events: [{ entity: 'PROMPT', action: 'CREATED' }],
        status: 'ACTIVE' as const,
      };
      mockFetchAPI.mockResolvedValue({} as any);

      await WebhooksApi.updateWebhook('wh-1', request);

      expect(mockFetchAPI).toHaveBeenCalledWith('ajax-api/2.0/mlflow/webhooks/wh-1', {
        method: 'PATCH',
        body: request,
      });
    });
  });

  describe('deleteWebhook', () => {
    it('calls fetchAPI with DELETE method', async () => {
      mockFetchAPI.mockResolvedValue(undefined as any);

      await WebhooksApi.deleteWebhook('wh-1');

      expect(mockFetchAPI).toHaveBeenCalledWith('ajax-api/2.0/mlflow/webhooks/wh-1', {
        method: 'DELETE',
      });
    });
  });

  describe('testWebhook', () => {
    it('calls fetchAPI with POST method and empty body', async () => {
      const mockResponse = { result: { success: true, response_status: 200 } };
      mockFetchAPI.mockResolvedValue(mockResponse as any);

      const result = await WebhooksApi.testWebhook('wh-1');

      expect(mockFetchAPI).toHaveBeenCalledWith('ajax-api/2.0/mlflow/webhooks/wh-1/test', {
        method: 'POST',
        body: {},
      });
      expect(result).toEqual(mockResponse);
    });

    it('propagates errors from fetchAPI', async () => {
      mockFetchAPI.mockRejectedValue(new Error('Network error'));

      await expect(WebhooksApi.testWebhook('wh-1')).rejects.toThrow('Network error');
    });
  });
});
