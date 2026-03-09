import { deleteJson, getJson, patchJson, postJson } from '../common/utils/FetchUtils';

export interface WebhookEvent {
  entity: string;
  action: string;
}

export interface Webhook {
  webhook_id: string;
  name: string;
  url: string;
  events: WebhookEvent[];
  status: string;
  creation_timestamp: string;
  last_updated_timestamp: string;
  description?: string;
}

export interface ListWebhooksResponse {
  webhooks: Webhook[];
}

export interface CreateWebhookRequest {
  name: string;
  url: string;
  events: WebhookEvent[];
  description?: string;
  secret?: string;
  status: 'ACTIVE' | 'DISABLED';
}

export interface UpdateWebhookRequest {
  name: string;
  url: string;
  events: WebhookEvent[];
  description?: string;
  secret?: string;
  status: 'ACTIVE' | 'DISABLED';
}

export interface TestWebhookResult {
  success: boolean;
  response_status?: number;
  error_message?: string;
}

export interface TestWebhookResponse {
  result: TestWebhookResult;
}

export const WebhooksApi = {
  listWebhooks: (): Promise<ListWebhooksResponse> => {
    return getJson({ relativeUrl: 'ajax-api/2.0/mlflow/webhooks' }) as Promise<ListWebhooksResponse>;
  },

  createWebhook: (request: CreateWebhookRequest): Promise<Webhook> => {
    return postJson({
      relativeUrl: 'ajax-api/2.0/mlflow/webhooks',
      data: request,
    }) as Promise<Webhook>;
  },

  updateWebhook: (webhookId: string, request: UpdateWebhookRequest): Promise<Webhook> => {
    return patchJson({
      relativeUrl: `ajax-api/2.0/mlflow/webhooks/${webhookId}`,
      data: request,
    }) as Promise<Webhook>;
  },

  deleteWebhook: (webhookId: string): Promise<void> => {
    return deleteJson({
      relativeUrl: `ajax-api/2.0/mlflow/webhooks/${webhookId}`,
    }) as Promise<void>;
  },

  testWebhook: (webhookId: string): Promise<TestWebhookResponse> => {
    return postJson({
      relativeUrl: `ajax-api/2.0/mlflow/webhooks/${webhookId}/test`,
      data: {},
    }) as Promise<TestWebhookResponse>;
  },
};
