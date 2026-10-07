import { useState } from 'react';
import { useIntl } from 'react-intl';
import { FormattedMessage } from 'react-intl';
import type { MCPAccessEndpoint } from '../types';
import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { useDeleteAccessEndpointMutation } from './useAccessEndpointMutation';

export const useDeleteAccessEndpointModal = ({ serverName }: { serverName: string }) => {
  const intl = useIntl();
  const [endpoint, setEndpoint] = useState<MCPAccessEndpoint | undefined>(undefined);
  const mutation = useDeleteAccessEndpointMutation();

  const DeleteAccessEndpointModal = (
    <ConfirmationModal
      componentId="mlflow.mcp_registry.detail.delete_endpoint_modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete access endpoint',
        description: 'Access endpoint delete confirmation modal title',
      })}
      visible={Boolean(endpoint)}
      message={
        <FormattedMessage
          defaultMessage="Are you sure you want to delete this access endpoint? This action cannot be undone."
          description="Access endpoint delete confirmation message"
        />
      }
      isLoading={mutation.isLoading}
      error={mutation.error?.message ?? null}
      onConfirm={() => {
        if (endpoint) {
          mutation.mutate({ serverName, endpointId: endpoint.id }, { onSuccess: () => setEndpoint(undefined) });
        }
      }}
      onCancel={() => {
        mutation.reset();
        setEndpoint(undefined);
      }}
    />
  );

  return { DeleteAccessEndpointModal, openDeleteEndpoint: (b: MCPAccessEndpoint) => setEndpoint(b) };
};
