import { FormattedMessage } from 'react-intl';
import { DeleteConfirmationModal } from '../common';
import { useDeleteSecret } from '../../hooks/useDeleteSecret';
import type { SecretInfo, ModelDefinition } from '../../types';

interface DeleteApiKeyModalProps {
  open: boolean;
  secret: SecretInfo | null;
  modelDefinitions: ModelDefinition[];
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteApiKeyModal = ({ open, secret, modelDefinitions, onClose, onSuccess }: DeleteApiKeyModalProps) => {
  const { mutateAsync: deleteSecret } = useDeleteSecret();

  const modelCount = modelDefinitions.length;
  const hasModels = modelCount > 0;

  const handleConfirm = async () => {
    if (!secret) return;
    await deleteSecret(secret.secret_id);
    onSuccess?.();
  };

  if (!secret) return null;

  return (
    <DeleteConfirmationModal
      open={open}
      onClose={onClose}
      onConfirm={handleConfirm}
      title="Delete API Key"
      itemName={secret.secret_name}
      itemType="API key"
      componentIdPrefix="mlflow.gateway.delete-api-key-modal"
      requireConfirmation={hasModels}
      warningMessage={
        hasModels ? (
          <FormattedMessage
            defaultMessage="This key is currently used by {modelCount, plural, one {# model definition} other {# model definitions}}. After deletion, you will need to attach a different API key to {modelCount, plural, one {this model} other {these models}} via the Edit Endpoint page."
            description="Warning about models using this key"
            values={{ modelCount }}
          />
        ) : undefined
      }
    />
  );
};
