import { useCallback, useState } from 'react';
import {
  Alert,
  Button,
  Drawer,
  Input,
  KeyIcon,
  PencilIcon,
  Spacer,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SecretDetails } from '../secrets';
import { SecretFormFields } from '../secrets';
import { useEditApiKeyModal } from '../../hooks/useEditApiKeyModal';
import { formatProviderName } from '../../utils/providerUtils';
import type { SecretInfo } from '../../types';

interface ApiKeyDetailsDrawerProps {
  open: boolean;
  secret: SecretInfo | null;
  onClose: () => void;
  onEditSuccess?: () => void;
}

export const ApiKeyDetailsDrawer = ({ open, secret, onClose, onEditSuccess }: ApiKeyDetailsDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isEditing, setIsEditing] = useState(false);

  const handleCloseEdit = useCallback(() => {
    setIsEditing(false);
  }, []);

  const handleSaveSuccess = useCallback(() => {
    setIsEditing(false);
    onEditSuccess?.();
  }, [onEditSuccess]);

  const {
    formData,
    errors,
    isLoading: isSaving,
    errorMessage,
    isFormValid,
    isDirty,
    provider,
    handleFormDataChange,
    handleSubmit,
    resetForm,
  } = useEditApiKeyModal({ secret, onClose: handleCloseEdit, onSuccess: handleSaveSuccess });

  const handleOpenChange = useCallback(
    (isOpen: boolean) => {
      if (!isOpen) {
        setIsEditing(false);
        onClose();
      }
    },
    [onClose],
  );

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.api-key-details.drawer"
        width={480}
        title={
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
            }}
          >
            <span
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                background: theme.colors.actionDefaultBackgroundHover,
                padding: theme.spacing.xs,
                color: theme.colors.blue500,
                height: 'min-content',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <KeyIcon />
            </span>
            {isEditing ? (
              <FormattedMessage defaultMessage="Edit API Key" description="Title for the API key edit drawer" />
            ) : (
              <FormattedMessage defaultMessage="API Key Details" description="Title for the API key details drawer" />
            )}
          </span>
        }
      >
        <Spacer size="md" />
        {secret && !isEditing && (
          <div css={{ display: 'flex', flexDirection: 'column' }}>
            <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.xs }}>
              <Typography.Title level={3} css={{ margin: 0 }}>
                {secret.secret_name}
              </Typography.Title>
              <Button
                componentId="mlflow.gateway.api-key-details.drawer.edit-button"
                size="small"
                type="tertiary"
                icon={<PencilIcon css={{ fontSize: 14 }} />}
                onClick={() => setIsEditing(true)}
                css={{ padding: theme.spacing.xs, lineHeight: 1 }}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Edit API key',
                  description: 'Gateway > API key details drawer > Edit API key button aria label',
                })}
              />
            </div>
            <SecretDetails secret={secret} showCard={false} hideTitle />
          </div>
        )}
        {secret && isEditing && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            {errorMessage && (
              <Alert
                componentId="mlflow.gateway.api-key-details.drawer.edit-error"
                type="error"
                message={errorMessage}
                closable={false}
              />
            )}

            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Text bold>
                <FormattedMessage defaultMessage="Key Name" description="Key name label" />
              </Typography.Text>
              <Tooltip
                componentId="mlflow.gateway.api-key-details.drawer.name-tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Key name cannot be changed. Create a new key if a different name is needed.',
                  description: 'Tooltip explaining why key name field is disabled in drawer edit',
                })}
              >
                <span css={{ display: 'block' }}>
                  <Input
                    componentId="mlflow.gateway.api-key-details.drawer.edit-name"
                    value={secret.secret_name}
                    disabled
                    css={{ backgroundColor: theme.colors.actionDisabledBackground }}
                  />
                </span>
              </Tooltip>
            </div>

            {provider && (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold>
                  <FormattedMessage defaultMessage="Provider" description="Provider label" />
                </Typography.Text>
                <Tooltip
                  componentId="mlflow.gateway.api-key-details.drawer.provider-tooltip"
                  content={intl.formatMessage({
                    defaultMessage: 'Provider cannot be changed. Create a new key if a different provider is needed.',
                    description: 'Tooltip explaining why provider field is disabled in drawer edit',
                  })}
                >
                  <span css={{ display: 'block' }}>
                    <Input
                      componentId="mlflow.gateway.api-key-details.drawer.edit-provider"
                      value={formatProviderName(provider)}
                      disabled
                      css={{ backgroundColor: theme.colors.actionDisabledBackground }}
                    />
                  </span>
                </Tooltip>
              </div>
            )}

            <SecretFormFields
              provider={provider}
              value={formData}
              onChange={handleFormDataChange}
              errors={errors}
              componentId="mlflow.gateway.api-key-details.drawer.edit"
              hideNameField
              secretPlaceholders={secret.masked_values}
            />

            <div css={{ display: 'flex', gap: theme.spacing.sm }}>
              <Button
                componentId="mlflow.gateway.api-key-details.drawer.save-button"
                type="primary"
                onClick={handleSubmit}
                disabled={!isFormValid || isSaving}
                loading={isSaving}
              >
                <FormattedMessage defaultMessage="Save Changes" description="Save changes button text" />
              </Button>
              <Button
                componentId="mlflow.gateway.api-key-details.drawer.cancel-button"
                onClick={resetForm}
                disabled={isSaving || !isDirty}
              >
                <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
              </Button>
            </div>
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};
