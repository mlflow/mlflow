import { useCallback } from 'react';
import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEditKeyValueTagsModal } from '../../common/hooks/useEditKeyValueTagsModal';
import { diffCurrentAndNewTags } from '../../common/utils/TagUtils';
import { MCPRegistryApi } from '../api';
import { MCP_QUERY_KEYS, tagsRecordToArray } from '../utils';
import type { MCPServer } from '../types';

type MCPServerTagEntity = { name: string; tags?: { key: string; value: string }[] };

type UpdateTagsPayload = {
  serverName: string;
  toAdd: { key: string; value: string }[];
  toDelete: { key: string }[];
};

export const useUpdateMCPServerTags = () => {
  const queryClient = useQueryClient();

  const updateMutation = useMutation<unknown, Error, UpdateTagsPayload>({
    mutationFn: async ({ serverName, toAdd, toDelete }) =>
      Promise.all([
        ...toAdd.map(({ key, value }) => MCPRegistryApi.setMCPServerTag(serverName, { key, value })),
        ...toDelete.map(({ key }) => MCPRegistryApi.deleteMCPServerTag(serverName, key)),
      ]),
    onSuccess: (_data, { serverName }) => {
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER, serverName]);
    },
  });

  const { EditTagsModal, showEditTagsModal } = useEditKeyValueTagsModal<MCPServerTagEntity>({
    valueRequired: true,
    saveTagsHandler: (entity, currentTags, newTags) => {
      const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);
      return new Promise<void>((resolve, reject) => {
        updateMutation.mutate(
          { serverName: entity.name, toAdd: addedOrModifiedTags, toDelete: deletedTags },
          { onSuccess: () => resolve(), onError: reject },
        );
      });
    },
  });

  const showEditServerTagsModal = useCallback(
    (server: MCPServer) => {
      showEditTagsModal({
        name: server.name,
        tags: tagsRecordToArray(server.tags),
      });
    },
    [showEditTagsModal],
  );

  return { EditTagsModal, showEditServerTagsModal };
};
