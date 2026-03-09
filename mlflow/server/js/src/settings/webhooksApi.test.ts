import { jest, describe, it, expect, beforeEach } from '@jest/globals';

jest.mock('../common/utils/FetchUtils', () => ({
  getJson: jest.fn(() => Promise.resolve()),
  postJson: jest.fn(() => Promise.resolve()),
  patchJson: jest.fn(() => Promise.resolve()),
  deleteJson: jest.fn(() => Promise.resolve()),
}));

import { getJson, postJson, patchJson, deleteJson } from '../common/utils/FetchUtils';
import { WebhooksApi } from './webhooksApi';

const mockGetJson = getJson as jest.MockedFunction<typeof getJson>;
const mockPostJson = postJson as jest.MockedFunction<typeof postJson>;
const mockPatchJson = patchJson as jest.MockedFunction<typeof patchJson>;
const mockDeleteJson = deleteJson as jest.MockedFunction<typeof deleteJson>;

describe('WebhooksApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('listWebhooks', () => {
    it('calls getJson with the correct URL', async () => {
      const mockResponse = { webhooks: [{ webhook_id: 'wh-1', name: 'Test' }] };
      mockGetJson.mockResolvedValue(mockResponse as any);

      const result = await WebhooksApi.listWebhooks();

      expect(mockGetJson).toHaveBeenCalledWith({
        relativeUrl: 'ajax-api/2.0/mlflow/webhooks',
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('createWebhook', () => {
    it('calls postJson with the correct URL and data', async () => {
      const request = {
        name: 'New Webhook',
        url: 'https://example.com/hook',
        events: [{ entity: 'PROMPT', action: 'CREATED' }],
        status: 'ACTIVE' as const,
      };
      mockPostJson.mockResolvedValue({} as any);

      await WebhooksApi.createWebhook(request);

      expect(mockPostJson).toHaveBeenCalledWith({
        relativeUrl: 'ajax-api/2.0/mlflow/webhooks',
        data: request,
      });
    });
  });

  describe('updateWebhook', () => {
    it('calls patchJson with the correct URL and data', async () => {
      const request = {
        name: 'Updated Webhook',
        url: 'https://example.com/hook',
        events: [{ entity: 'PROMPT', action: 'CREATED' }],
        status: 'ACTIVE' as const,
      };
      mockPatchJson.mockResolvedValue({} as any);

      await WebhooksApi.updateWebhook('wh-1', request);

      expect(mockPatchJson).toHaveBeenCalledWith({
        relativeUrl: 'ajax-api/2.0/mlflow/webhooks/wh-1',
        data: request,
      });
    });
  });

  describe('deleteWebhook', () => {
    it('calls deleteJson with the correct URL', async () => {
      mockDeleteJson.mockResolvedValue(undefined as any);

      await WebhooksApi.deleteWebhook('wh-1');

      expect(mockDeleteJson).toHaveBeenCalledWith({
        relativeUrl: 'ajax-api/2.0/mlflow/webhooks/wh-1',
      });
    });
  });

  describe('testWebhook', () => {
    it('calls postJson with the correct URL and empty data', async () => {
      const mockResponse = { result: { success: true, response_status: 200 } };
      mockPostJson.mockResolvedValue(mockResponse as any);

      const result = await WebhooksApi.testWebhook('wh-1');

      expect(mockPostJson).toHaveBeenCalledWith({
        relativeUrl: 'ajax-api/2.0/mlflow/webhooks/wh-1/test',
        data: {},
      });
      expect(result).toEqual(mockResponse);
    });

    it('propagates errors from postJson', async () => {
      mockPostJson.mockRejectedValue(new Error('Network error'));

      await expect(WebhooksApi.testWebhook('wh-1')).rejects.toThrow('Network error');
    });
  });
});
