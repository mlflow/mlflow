/**
 * React Query hooks for labeling sessions, schemas, and items
 */

import { useQuery, useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { LabelingApi } from './LabelingApi';
import type {
  CreateLabelingSessionRequest,
  CreateLabelingSchemaRequest,
  CreateLabelingSessionItemsRequest,
  UpdateLabelingSessionRequest,
  UpdateLabelingSessionItemRequest,
  DeleteLabelingSessionItemsRequest,
  ListLabelingSessionItemsRequest,
} from './LabelingApi';

/**
 * Query key factories for cache management
 */
export const labelingQueryKeys = {
  all: ['labeling'] as const,
  sessions: (experimentId?: string) => ['labeling', 'sessions', experimentId] as const,
  session: (sessionId: string) => ['labeling', 'session', sessionId] as const,
  schemas: (sessionId: string) => ['labeling', 'schemas', sessionId] as const,
  schema: (sessionId: string, name: string) => ['labeling', 'schema', sessionId, name] as const,
  items: (sessionId: string) => ['labeling', 'items', sessionId] as const,
  item: (itemId: string) => ['labeling', 'item', itemId] as const,
};

// ===== Labeling Sessions Hooks =====

export const useGetLabelingSessions = (experimentId: string, options?: { enabled?: boolean }) => {
  return useQuery(labelingQueryKeys.sessions(experimentId), {
    queryFn: () => LabelingApi.listLabelingSessions(experimentId),
    enabled: options?.enabled !== false,
  });
};

export const useGetLabelingSession = (sessionId: string, options?: { enabled?: boolean }) => {
  return useQuery(labelingQueryKeys.session(sessionId), {
    queryFn: () => LabelingApi.getLabelingSession(sessionId),
    enabled: options?.enabled !== false && Boolean(sessionId),
  });
};

export const useCreateLabelingSession = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateLabelingSessionRequest) => LabelingApi.createLabelingSession(request),
    onSuccess: (data) => {
      // Invalidate sessions list for the experiment
      queryClient.invalidateQueries(labelingQueryKeys.sessions(data.labeling_session.experiment_id));
    },
  });
};

export const useUpdateLabelingSession = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: UpdateLabelingSessionRequest) => LabelingApi.updateLabelingSession(request),
    onSuccess: (data, variables) => {
      // Invalidate the specific session
      queryClient.invalidateQueries(labelingQueryKeys.session(variables.labeling_session_id));
      // Invalidate sessions list
      queryClient.invalidateQueries(labelingQueryKeys.sessions(data.labeling_session.experiment_id));
    },
  });
};

export const useDeleteLabelingSession = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => LabelingApi.deleteLabelingSession(sessionId),
    onSuccess: (_, sessionId) => {
      // Invalidate the specific session
      queryClient.invalidateQueries(labelingQueryKeys.session(sessionId));
      // Invalidate all sessions lists (we don't know which experiment)
      queryClient.invalidateQueries(labelingQueryKeys.sessions());
    },
  });
};

// ===== Labeling Schemas Hooks =====

export const useGetLabelingSchemas = (sessionId: string, options?: { enabled?: boolean }) => {
  return useQuery(labelingQueryKeys.schemas(sessionId), {
    queryFn: () => LabelingApi.listLabelingSchemas(sessionId),
    enabled: options?.enabled !== false && Boolean(sessionId),
  });
};

export const useGetLabelingSchema = (
  sessionId: string,
  name: string,
  options?: { enabled?: boolean },
) => {
  return useQuery(labelingQueryKeys.schema(sessionId, name), {
    queryFn: () => LabelingApi.getLabelingSchema(sessionId, name),
    enabled: options?.enabled !== false && Boolean(sessionId) && Boolean(name),
  });
};

export const useCreateLabelingSchema = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateLabelingSchemaRequest) => LabelingApi.createLabelingSchema(request),
    onSuccess: (data, variables) => {
      // Invalidate schemas list for the session
      queryClient.invalidateQueries(labelingQueryKeys.schemas(variables.labeling_session_id));
      // Invalidate the session (it contains schemas)
      queryClient.invalidateQueries(labelingQueryKeys.session(variables.labeling_session_id));
    },
  });
};

export const useDeleteLabelingSchema = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, name }: { sessionId: string; name: string }) =>
      LabelingApi.deleteLabelingSchema(sessionId, name),
    onSuccess: (_, variables) => {
      // Invalidate the specific schema
      queryClient.invalidateQueries(labelingQueryKeys.schema(variables.sessionId, variables.name));
      // Invalidate schemas list
      queryClient.invalidateQueries(labelingQueryKeys.schemas(variables.sessionId));
      // Invalidate the session
      queryClient.invalidateQueries(labelingQueryKeys.session(variables.sessionId));
    },
  });
};

// ===== Labeling Session Items Hooks =====

export const useGetLabelingItems = (
  request: ListLabelingSessionItemsRequest,
  options?: { enabled?: boolean },
) => {
  return useQuery(labelingQueryKeys.items(request.labeling_session_id), {
    queryFn: () => LabelingApi.listLabelingSessionItems(request),
    enabled: options?.enabled !== false && Boolean(request.labeling_session_id),
  });
};

export const useGetLabelingItem = (itemId: string, options?: { enabled?: boolean }) => {
  return useQuery(labelingQueryKeys.item(itemId), {
    queryFn: () => LabelingApi.getLabelingSessionItem(itemId),
    enabled: options?.enabled !== false && Boolean(itemId),
  });
};

export const useCreateLabelingItems = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateLabelingSessionItemsRequest) =>
      LabelingApi.createLabelingSessionItems(request),
    onSuccess: (_, variables) => {
      // Invalidate items list for the session
      queryClient.invalidateQueries(labelingQueryKeys.items(variables.labeling_session_id));
    },
  });
};

export const useUpdateLabelingItem = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: UpdateLabelingSessionItemRequest) =>
      LabelingApi.updateLabelingSessionItem(request),
    onSuccess: (data, variables) => {
      // Invalidate the specific item
      queryClient.invalidateQueries(labelingQueryKeys.item(variables.labeling_item_id));
      // Invalidate items list
      queryClient.invalidateQueries(labelingQueryKeys.items(data.labeling_item.labeling_session_id));
    },
  });
};

export const useDeleteLabelingItems = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: DeleteLabelingSessionItemsRequest) =>
      LabelingApi.deleteLabelingSessionItems(request),
    onSuccess: (_, variables) => {
      // Invalidate specific items
      variables.labeling_item_ids.forEach((itemId) => {
        queryClient.invalidateQueries(labelingQueryKeys.item(itemId));
      });
      // Invalidate items list
      queryClient.invalidateQueries(labelingQueryKeys.items(variables.labeling_session_id));
    },
  });
};
