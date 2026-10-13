import { fetchAPI, getAjaxUrl, HTTPMethods } from '../common/utils/FetchUtils';

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
    return fetchAPI(getAjaxUrl('ajax-api/2.0/mlflow/webhooks')) as Promise<ListWebhooksResponse>;
  },

  createWebhook: (request: CreateWebhookRequest): Promise<Webhook> => {
    return fetchAPI(getAjaxUrl('ajax-api/2.0/mlflow/webhooks'), {
      method: HTTPMethods.POST,
      body: request,
    }) as Promise<Webhook>;
  },

  updateWebhook: (webhookId: string, request: UpdateWebhookRequest): Promise<Webhook> => {
    return fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/webhooks/${webhookId}`), {
      method: HTTPMethods.PATCH,
      body: request,
    }) as Promise<Webhook>;
  },

  deleteWebhook: (webhookId: string): Promise<void> => {
    return fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/webhooks/${webhookId}`), {
      method: HTTPMethods.DELETE,
    }) as Promise<void>;
  },

  testWebhook: (webhookId: string): Promise<TestWebhookResponse> => {
    return fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/webhooks/${webhookId}/test`), {
      method: HTTPMethods.POST,
      body: {},
    }) as Promise<TestWebhookResponse>;
  },
};
