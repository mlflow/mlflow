import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { DeleteConfirmationModal } from '../common';
import { useDeleteSecret } from '../../hooks/useDeleteSecret';
import type { SecretInfo, Endpoint } from '../../types';

interface DeleteApiKeyModalProps {
  open: boolean;
  secret: SecretInfo | null;
  endpoints: Endpoint[];
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteApiKeyModal = ({ open, secret, endpoints, onClose, onSuccess }: DeleteApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const { mutateAsync: deleteSecret } = useDeleteSecret();

  const hasEndpoints = endpoints.length > 0;

  const handleConfirm = async () => {
    if (!secret) return;
    await deleteSecret(secret.secret_id);
    onSuccess?.();
  };

  if (!secret) return null;

  const renderEndpointsList = () => {
    if (!hasEndpoints) return undefined;

    return (
      <div>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Endpoints using this key ({count})"
            description="Gateway > Delete API key modal > Endpoints list header"
            values={{ count: endpoints.length }}
          />
        </Typography.Text>
        <div
          css={{
            marginTop: theme.spacing.xs,
            maxHeight: 120,
            overflowY: 'auto',
            border: `1px solid ${theme.colors.borderDecorative}`,
            borderRadius: theme.general.borderRadiusBase,
          }}
        >
          {endpoints.map((endpoint) => {
            const modelNames = endpoint.model_mappings
              ?.map((mapping) => mapping.model_definition?.model_name)
              .filter(Boolean)
              .join(', ');

            return (
              <div
                key={endpoint.endpoint_id}
                css={{
                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                  borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                  '&:last-child': { borderBottom: 'none' },
                }}
              >
                <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                  {endpoint.name || endpoint.endpoint_id}
                </Typography.Text>
                {modelNames && (
                  <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
                    {modelNames}
                  </Typography.Text>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <DeleteConfirmationModal
      open={open}
      onClose={onClose}
      onConfirm={handleConfirm}
      title="Delete API Key"
      itemName={secret.secret_name}
      itemType="API key"
      componentIdPrefix="mlflow.gateway.delete-api-key-modal"
      requireConfirmation={hasEndpoints}
      warningMessage={
        hasEndpoints ? (
          <FormattedMessage
            defaultMessage="This key is currently in use. After deleting, you will need to attach a different API key to continue using the endpoints currently using this key."
            description="Gateway > Delete API key modal > Warning about endpoints using this key"
          />
        ) : undefined
      }
      additionalContent={renderEndpointsList()}
    />
  );
};
