import { FormattedMessage } from 'react-intl';
import { DeleteConfirmationModal } from '../common';
import { useDeleteEndpoint } from '../../hooks/useDeleteEndpoint';
import type { Endpoint, EndpointBinding } from '../../types';

interface DeleteEndpointModalProps {
  open: boolean;
  endpoint: Endpoint | null;
  bindings: EndpointBinding[];
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteEndpointModal = ({ open, endpoint, bindings, onClose, onSuccess }: DeleteEndpointModalProps) => {
  const { mutateAsync: deleteEndpoint } = useDeleteEndpoint();

  const bindingCount = bindings.length;
  const hasBindings = bindingCount > 0;

  const handleConfirm = async () => {
    if (!endpoint) return;
    await deleteEndpoint(endpoint.endpoint_id);
    onSuccess?.();
  };

  if (!endpoint) return null;

  return (
    <DeleteConfirmationModal
      open={open}
      onClose={onClose}
      onConfirm={handleConfirm}
      title="Delete Endpoint"
      itemName={endpoint.name}
      itemType="endpoint"
      componentIdPrefix="mlflow.gateway.delete-endpoint-modal"
      requireConfirmation={hasBindings}
      warningMessage={
        hasBindings ? (
          <FormattedMessage
            defaultMessage="This endpoint is currently used by {bindingCount, plural, one {# resource} other {# resources}}. Deleting it will break those connections."
            description="Warning about resources using this endpoint"
            values={{ bindingCount }}
          />
        ) : undefined
      }
    />
  );
};
