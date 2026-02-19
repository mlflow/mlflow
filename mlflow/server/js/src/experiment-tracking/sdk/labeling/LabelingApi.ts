/**
 * API client for labeling sessions, schemas, and items
 */

import { fetchEndpoint } from '../../../common/utils/FetchUtils';
import type {
  LabelingSession,
  LabelingSchema,
  LabelingItem,
  LabelingItemStatus,
  LabelingSchemaValueType,
} from '../../types/labeling';

/**
 * API request/response types
 */
export interface CreateLabelingSessionRequest {
  experiment_id: string;
  name: string;
}

export interface CreateLabelingSessionResponse {
  labeling_session: LabelingSession;
}

export interface GetLabelingSessionResponse {
  labeling_session: LabelingSession;
}

export interface ListLabelingSessionsResponse {
  labeling_sessions: LabelingSession[];
}

export interface UpdateLabelingSessionRequest {
  labeling_session_id: string;
  name: string;
}

export interface UpdateLabelingSessionResponse {
  labeling_session: LabelingSession;
}

export interface CreateLabelingSchemaRequest {
  labeling_session_id: string;
  name: string;
  type: 'FEEDBACK' | 'EXPECTATION';
  title: string;
  schema: LabelingSchemaValueType;
  instructions?: string;
}

export interface CreateLabelingSchemaResponse {
  labeling_schema: LabelingSchema;
}

export interface GetLabelingSchemaResponse {
  labeling_schema: LabelingSchema;
}

export interface ListLabelingSchemasResponse {
  labeling_schemas: LabelingSchema[];
}

export interface CreateLabelingSessionItemsRequest {
  labeling_session_id: string;
  items: Array<{
    trace_id?: string;
    dataset_record_id?: string;
    dataset_id?: string;
  }>;
}

export interface CreateLabelingSessionItemsResponse {
  labeling_items: LabelingItem[];
}

export interface GetLabelingSessionItemResponse {
  labeling_item: LabelingItem;
}

export interface ListLabelingSessionItemsRequest {
  labeling_session_id: string;
  page_token?: string;
  max_results?: number;
}

export interface ListLabelingSessionItemsResponse {
  labeling_items: LabelingItem[];
  next_page_token?: string;
}

export interface UpdateLabelingSessionItemRequest {
  labeling_item_id: string;
  state: LabelingItemStatus;
}

export interface UpdateLabelingSessionItemResponse {
  labeling_item: LabelingItem;
}

export interface DeleteLabelingSessionItemsRequest {
  labeling_session_id: string;
  labeling_item_ids: string[];
}

/**
 * Labeling API client
 */
export const LabelingApi = {
  // ===== Labeling Sessions =====

  createLabelingSession: (request: CreateLabelingSessionRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/labeling/sessions',
      method: 'POST',
      body: JSON.stringify(request),
    }) as Promise<CreateLabelingSessionResponse>;
  },

  getLabelingSession: (labelingSessionId: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${labelingSessionId}`,
      method: 'GET',
    }) as Promise<GetLabelingSessionResponse>;
  },

  listLabelingSessions: (experimentId: string) => {
    const params = new URLSearchParams({ experiment_id: experimentId });
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions?${params.toString()}`,
      method: 'GET',
    }) as Promise<ListLabelingSessionsResponse>;
  },

  updateLabelingSession: (request: UpdateLabelingSessionRequest) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${request.labeling_session_id}`,
      method: 'PATCH',
      body: JSON.stringify(request),
    }) as Promise<UpdateLabelingSessionResponse>;
  },

  deleteLabelingSession: (labelingSessionId: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${labelingSessionId}`,
      method: 'DELETE',
    }) as Promise<void>;
  },

  // ===== Labeling Schemas =====

  createLabelingSchema: (request: CreateLabelingSchemaRequest) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${request.labeling_session_id}/schemas`,
      method: 'POST',
      body: JSON.stringify(request),
    }) as Promise<CreateLabelingSchemaResponse>;
  },

  getLabelingSchema: (labelingSessionId: string, name: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${labelingSessionId}/schemas/${encodeURIComponent(name)}`,
      method: 'GET',
    }) as Promise<GetLabelingSchemaResponse>;
  },

  listLabelingSchemas: (labelingSessionId: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${labelingSessionId}/schemas`,
      method: 'GET',
    }) as Promise<ListLabelingSchemasResponse>;
  },

  deleteLabelingSchema: (labelingSessionId: string, name: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${labelingSessionId}/schemas/${encodeURIComponent(name)}`,
      method: 'DELETE',
    }) as Promise<void>;
  },

  // ===== Labeling Session Items =====

  createLabelingSessionItems: (request: CreateLabelingSessionItemsRequest) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${request.labeling_session_id}/items`,
      method: 'POST',
      body: JSON.stringify(request),
    }) as Promise<CreateLabelingSessionItemsResponse>;
  },

  getLabelingSessionItem: (labelingItemId: string) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/items/${labelingItemId}`,
      method: 'GET',
    }) as Promise<GetLabelingSessionItemResponse>;
  },

  listLabelingSessionItems: (request: ListLabelingSessionItemsRequest) => {
    const params = new URLSearchParams();
    if (request.page_token) {
      params.append('page_token', request.page_token);
    }
    if (request.max_results) {
      params.append('max_results', request.max_results.toString());
    }
    const queryString = params.toString();
    const url = `ajax-api/3.0/mlflow/labeling/sessions/${request.labeling_session_id}/items${queryString ? `?${queryString}` : ''}`;
    return fetchEndpoint({
      relativeUrl: url,
      method: 'GET',
    }) as Promise<ListLabelingSessionItemsResponse>;
  },

  updateLabelingSessionItem: (request: UpdateLabelingSessionItemRequest) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/items/${request.labeling_item_id}`,
      method: 'PATCH',
      body: JSON.stringify(request),
    }) as Promise<UpdateLabelingSessionItemResponse>;
  },

  deleteLabelingSessionItems: (request: DeleteLabelingSessionItemsRequest) => {
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/labeling/sessions/${request.labeling_session_id}/items`,
      method: 'DELETE',
      body: JSON.stringify(request),
    }) as Promise<void>;
  },
};
