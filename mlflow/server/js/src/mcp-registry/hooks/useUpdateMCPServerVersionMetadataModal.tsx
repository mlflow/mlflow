import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEditKeyValueTagsModal } from '../../common/hooks/useEditKeyValueTagsModal';
import { MCPRegistryApi } from '../api';
import type { MCPServerVersion } from '../types';
import { MCP_QUERY_KEYS, tagsRecordToArray } from '../utils';
import { useCallback } from 'react';
import { diffCurrentAndNewTags } from '../../common/utils/TagUtils';
import { FormattedMessage } from 'react-intl';
import type { KeyValueEntity } from '../../common/types';

type UpdateMCPServerVersionMetadataPayload = {
  serverName: string;
  version: string;
  toAdd: { key: string; value: string }[];
  toDelete: { key: string }[];
};

export const useUpdateMCPServerVersionMetadataModal = ({ serverName }: { serverName: string }) => {
  const queryClient = useQueryClient();

  const updateMutation = useMutation<unknown, Error, UpdateMCPServerVersionMetadataPayload>({
    mutationFn: async ({ serverName: name, version, toAdd, toDelete }) => {
      return Promise.all([
        ...toAdd.map(({ key, value }) => MCPRegistryApi.setMCPServerVersionTag(name, version, { key, value })),
        ...toDelete.map(({ key }) => MCPRegistryApi.deleteMCPServerVersionTag(name, version, key)),
      ]);
    },
  });

  const {
    EditTagsModal: EditMCPServerVersionMetadataModal,
    showEditTagsModal,
    isLoading,
  } = useEditKeyValueTagsModal<{ version: string; tags?: KeyValueEntity[] }>({
    title: (
      <FormattedMessage
        defaultMessage="Add/Edit Metadata"
        description="Title for a modal that allows the user to add or edit metadata tags on MCP server versions."
      />
    ),
    valueRequired: true,
    saveTagsHandler: (editedVersion, currentTags, newTags) => {
      const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);

      return new Promise<void>((resolve, reject) => {
        updateMutation.mutate(
          {
            serverName,
            version: editedVersion.version,
            toAdd: addedOrModifiedTags,
            toDelete: deletedTags,
          },
          {
            onSuccess: () => {
              queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_VERSIONS, serverName]);
              queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER, serverName]);
              queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER_LATEST_VERSION, serverName]);
              queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
              resolve();
            },
            onError: reject,
          },
        );
      });
    },
  });

  const showEditMetadataModal = useCallback(
    (version: MCPServerVersion) =>
      showEditTagsModal({
        version: version.version,
        tags: tagsRecordToArray(version.tags),
      }),
    [showEditTagsModal],
  );

  return { EditMCPServerVersionMetadataModal, showEditMetadataModal, isLoading };
};
