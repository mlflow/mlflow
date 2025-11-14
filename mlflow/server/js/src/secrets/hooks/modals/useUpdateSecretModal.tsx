import { FormUI, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState, useCallback } from 'react';
import type { Secret } from '../../types';
import { useUpdateSecretMutation } from '../useUpdateSecretMutation';
import { SecretBindingsList } from '../../components/SecretBindingsList';

export interface UseUpdateSecretModalProps {
  secret: Secret | null;
  onSuccess?: (secretName: string) => void;
}

interface UseUpdateSecretModalReturn {
  UpdateSecretModal: JSX.Element | null;
  openModal: () => void;
}

export const useUpdateSecretModal = ({ secret, onSuccess }: UseUpdateSecretModalProps): UseUpdateSecretModalReturn => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [open, setOpen] = useState(false);
  const [secretValue, setSecretValue] = useState('');
  const [error, setError] = useState<string>();

  const { updateSecret, isLoading } = useUpdateSecretMutation({
    onSuccess: () => {
      const secretName = secret?.secret_name || '';
      onSuccess?.(secretName);
      setSecretValue('');
      setError(undefined);
      setOpen(false);
    },
    onError: (err: Error) => {
      setError(
        err.message ||
          intl.formatMessage({
            defaultMessage: 'Failed to update secret. Please try again.',
            description: 'Update secret modal > update failed error message',
          }),
      );
    },
  });

  const handleUpdate = useCallback(() => {
    if (!secret) return;

    if (!secretValue) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Secret value is required',
          description: 'Update secret modal > secret value required validation',
        }),
      );
      return;
    }

    updateSecret({
      secret_id: secret.secret_id,
      secret_value: secretValue,
    });
  }, [secret, secretValue, updateSecret, intl]);

  const handleCancel = useCallback(() => {
    setSecretValue('');
    setError(undefined);
    setOpen(false);
  }, []);

  const modalElement = secret ? (
    <Modal
      componentId="mlflow.secrets.update_secret_modal"
      visible={open}
      onCancel={handleCancel}
      okText={intl.formatMessage({
        defaultMessage: 'Update',
        description: 'Update secret modal > update button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Update secret modal > cancel button text',
      })}
      onOk={handleUpdate}
      okButtonProps={{ loading: isLoading, disabled: !secretValue }}
      title={<FormattedMessage defaultMessage="Update Secret Value" description="Update secret modal > modal title" />}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <FormUI.Label>
            <FormattedMessage defaultMessage="Secret Name" description="Update secret modal > secret name label" />
          </FormUI.Label>
          <Typography.Text bold>{secret.secret_name}</Typography.Text>
        </div>

        <div>
          <SecretBindingsList secretId={secret.secret_id} variant="warning" isSharedSecret={secret.is_shared} />
        </div>

        <div>
          <FormUI.Label htmlFor="update-secret-value-input">
            <FormattedMessage
              defaultMessage="New Secret Value"
              description="Update secret modal > new secret value label"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.update_secret_modal.value"
            id="update-secret-value-input"
            type="password"
            autoComplete="off"
            data-form-type="other"
            data-lpignore="true"
            data-1p-ignore="true"
            data-bwignore="true"
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter new secret value',
              description: 'Update secret modal > secret value placeholder',
            })}
            value={secretValue}
            onChange={(e) => {
              setSecretValue(e.target.value);
              setError(undefined);
            }}
          />
          {error && <FormUI.Message type="error" message={error} />}
        </div>
      </div>
    </Modal>
  ) : null;

  return {
    UpdateSecretModal: modalElement,
    openModal: () => {
      setSecretValue('');
      setError(undefined);
      setOpen(true);
    },
  };
};
