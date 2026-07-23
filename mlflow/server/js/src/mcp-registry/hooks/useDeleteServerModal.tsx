import { useState } from 'react';
import { useIntl } from 'react-intl';
import { FormattedMessage } from 'react-intl';
import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { useDeleteMCPServer } from './useMCPServerVersionMutations';

export const useDeleteServerModal = ({ serverName, onDeleted }: { serverName: string; onDeleted: () => void }) => {
  const intl = useIntl();
  const [visible, setVisible] = useState(false);
  const mutation = useDeleteMCPServer();

  const DeleteServerModal = (
    <ConfirmationModal
      componentId="mlflow.mcp_registry.detail.delete_server_modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete MCP server',
        description: 'MCP server delete confirmation modal title',
      })}
      visible={visible}
      message={
        <FormattedMessage
          defaultMessage="Are you sure you want to delete this MCP server and all its versions? This action cannot be undone."
          description="MCP server delete confirmation message"
        />
      }
      isLoading={mutation.isLoading}
      error={mutation.error?.message ?? null}
      onConfirm={() => {
        mutation.mutate(serverName, {
          onSuccess: () => {
            setVisible(false);
            onDeleted();
          },
        });
      }}
      onCancel={() => {
        mutation.reset();
        setVisible(false);
      }}
    />
  );

  return { DeleteServerModal, openDeleteModal: () => setVisible(true) };
};
