import { FormattedMessage } from 'react-intl';
import { DeleteConfirmationModal } from '../common';
import { useDeleteEndpointMutation } from '../../hooks/useDeleteEndpointMutation';
import type { Endpoint, EndpointBinding } from '../../types';

interface DeleteEndpointModalProps {
  open: boolean;
  endpoint: Endpoint | null;
  bindings: EndpointBinding[];
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteEndpointModal = ({ open, endpoint, bindings, onClose, onSuccess }: DeleteEndpointModalProps) => {
  const { mutateAsync: deleteEndpoint } = useDeleteEndpointMutation();

  const bindingCount = bindings.length;
  const hasBindings = bindingCount > 0;
  const endpointName = endpoint?.name ?? endpoint?.endpoint_id ?? '';

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
      itemName={endpointName}
      itemType="endpoint"
      componentIdPrefix="mlflow.gateway.delete-endpoint-modal"
      requireConfirmation={hasBindings}
      warningMessage={
        hasBindings ? (
          <FormattedMessage
            defaultMessage="This endpoint is currently bound to {bindingCount, plural, one {# resource} other {# resources}}. After deletion, these bindings will be removed."
            description="Warning about resources bound to this endpoint"
            values={{ bindingCount }}
          />
        ) : undefined
      }
    />
  );
};
