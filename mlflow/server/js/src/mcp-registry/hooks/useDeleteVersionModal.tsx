import { useState } from 'react';
import { useIntl } from 'react-intl';
import { FormattedMessage } from 'react-intl';
import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { useDeleteMCPServerVersion } from './useMCPServerVersionMutations';

export const useDeleteVersionModal = ({ serverName }: { serverName: string }) => {
  const intl = useIntl();
  const [versionToDelete, setVersionToDelete] = useState<string | undefined>(undefined);
  const mutation = useDeleteMCPServerVersion(serverName);

  const DeleteVersionModal = (
    <ConfirmationModal
      componentId="mlflow.mcp_registry.detail.delete_version_modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete version',
        description: 'MCP server delete version confirmation modal title',
      })}
      visible={Boolean(versionToDelete)}
      message={
        <FormattedMessage
          defaultMessage="Are you sure you want to delete version {version}? This action cannot be undone."
          description="MCP server delete version confirmation message"
          values={{ version: versionToDelete ?? '' }}
        />
      }
      isLoading={mutation.isLoading}
      error={mutation.error?.message ?? null}
      onConfirm={() => {
        if (versionToDelete) {
          mutation.mutate(versionToDelete, {
            onSuccess: () => setVersionToDelete(undefined),
          });
        }
      }}
      onCancel={() => {
        mutation.reset();
        setVersionToDelete(undefined);
      }}
    />
  );

  return { DeleteVersionModal, openDeleteVersionModal: (version: string) => setVersionToDelete(version) };
};
