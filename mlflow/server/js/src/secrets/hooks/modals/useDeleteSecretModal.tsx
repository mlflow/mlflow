import { DangerModal, FormUI, Input, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState, useCallback } from 'react';
import type { Secret } from '../../types';
import { useDeleteSecretMutation } from '../useDeleteSecretMutation';
import { SecretBindingsList } from '../../components/SecretBindingsList';

export interface UseDeleteSecretModalProps {
  secret: Secret | null;
  onSuccess?: () => void;
}

interface UseDeleteSecretModalReturn {
  DeleteSecretModal: JSX.Element | null;
  openModal: () => void;
}

export const useDeleteSecretModal = ({ secret, onSuccess }: UseDeleteSecretModalProps): UseDeleteSecretModalReturn => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [open, setOpen] = useState(false);
  const [confirmationText, setConfirmationText] = useState('');

  const { deleteSecret, isLoading } = useDeleteSecretMutation({
    onSuccess: () => {
      setConfirmationText('');
      setOpen(false);
      onSuccess?.();
    },
    onError: (error: Error) => {
      console.error('Failed to delete secret:', error);
    },
  });

  const handleDelete = useCallback(() => {
    if (!secret) return;
    deleteSecret({ secret_id: secret.secret_id });
  }, [secret, deleteSecret]);

  const handleCancel = useCallback(() => {
    setConfirmationText('');
    setOpen(false);
  }, []);

  const isDeleteDisabled = !secret || confirmationText !== secret.secret_name;

  const modalElement = secret ? (
    <DangerModal
      componentId="mlflow.secrets.delete_secret_modal"
      visible={open}
      onCancel={handleCancel}
      okText={intl.formatMessage({
        defaultMessage: 'Delete',
        description: 'Delete secret modal > delete button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Delete secret modal > cancel button text',
      })}
      onOk={handleDelete}
      okButtonProps={{ loading: isLoading, disabled: isDeleteDisabled }}
      title={<FormattedMessage defaultMessage="Delete Secret" description="Delete secret modal > modal title" />}
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Are you sure you want to delete the secret '{secretName}'? This action cannot be undone."
            description="Delete secret modal > confirmation message"
            values={{ secretName: secret.secret_name }}
          />
        </Typography.Text>

        <SecretBindingsList secretId={secret.secret_id} variant="warning" isSharedSecret={secret.is_shared} />

        <div>
          <FormUI.Label htmlFor="delete-secret-confirmation-input">
            <FormattedMessage
              defaultMessage="To confirm, type the secret name:"
              description="Delete secret modal > confirmation input label"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.delete_secret_modal.confirmation_input"
            id="delete-secret-confirmation-input"
            placeholder={secret.secret_name}
            value={confirmationText}
            onChange={(e) => setConfirmationText(e.target.value)}
            autoComplete="off"
          />
        </div>
      </div>
    </DangerModal>
  ) : null;

  return {
    DeleteSecretModal: modalElement,
    openModal: () => {
      setConfirmationText('');
      setOpen(true);
    },
  };
};
