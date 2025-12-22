import { Alert, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SecretFormFields } from '../secrets';
import { useEditApiKeyModal } from '../../hooks/useEditApiKeyModal';
import { formatProviderName } from '../../utils/providerUtils';
import type { SecretInfo } from '../../types';

interface EditApiKeyModalProps {
  open: boolean;
  secret: SecretInfo | null;
  onClose: () => void;
  onSuccess?: () => void;
}

export const EditApiKeyModal = ({ open, secret, onClose, onSuccess }: EditApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const {
    formData,
    errors,
    isLoading,
    errorMessage,
    isFormValid,
    provider,
    handleFormDataChange,
    handleSubmit,
    handleClose,
  } = useEditApiKeyModal({ secret, onClose, onSuccess });

  if (!secret) return null;

  return (
    <Modal
      componentId="mlflow.gateway.edit-api-key-modal"
      title={intl.formatMessage({
        defaultMessage: 'Edit API Key',
        description: 'Title for edit API key modal',
      })}
      visible={open}
      onCancel={handleClose}
      onOk={handleSubmit}
      okText={intl.formatMessage({
        defaultMessage: 'Save Changes',
        description: 'Save changes button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel button text',
      })}
      confirmLoading={isLoading}
      okButtonProps={{ disabled: !isFormValid }}
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {errorMessage && (
          <Alert
            componentId="mlflow.gateway.edit-api-key-modal.error"
            type="error"
            message={errorMessage}
            closable={false}
          />
        )}

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Key Name" description="Key name label" />
          </Typography.Text>
          <Input
            componentId="mlflow.gateway.edit-api-key-modal.name"
            value={secret.secret_name}
            disabled
            css={{ backgroundColor: theme.colors.actionDisabledBackground }}
          />
        </div>

        {provider && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Provider" description="Provider label" />
            </Typography.Text>
            <Input
              componentId="mlflow.gateway.edit-api-key-modal.provider"
              value={formatProviderName(provider)}
              disabled
              css={{ backgroundColor: theme.colors.actionDisabledBackground }}
            />
          </div>
        )}

        <SecretFormFields
          provider={provider}
          value={formData}
          onChange={handleFormDataChange}
          errors={errors}
          componentIdPrefix="mlflow.gateway.edit-api-key-modal"
          hideNameField
        />
      </div>
    </Modal>
  );
};
