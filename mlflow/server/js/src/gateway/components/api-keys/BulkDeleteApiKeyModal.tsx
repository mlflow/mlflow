import { useState } from 'react';
import { Alert, Button, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDeleteSecret } from '../../hooks/useDeleteSecret';
import type { SecretInfo, Endpoint } from '../../types';

interface BulkDeleteApiKeyModalProps {
  open: boolean;
  secrets: SecretInfo[];
  getEndpointsForSecret: (secretId: string) => Endpoint[];
  onClose: () => void;
  onSuccess: () => void;
}

export const BulkDeleteApiKeyModal = ({
  open,
  secrets,
  getEndpointsForSecret,
  onClose,
  onSuccess,
}: BulkDeleteApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { mutateAsync: deleteSecret } = useDeleteSecret();
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const secretsWithEndpoints = secrets.filter((s) => getEndpointsForSecret(s.secret_id).length > 0);
  const hasEndpoints = secretsWithEndpoints.length > 0;

  const handleDelete = async () => {
    setIsDeleting(true);
    setError(null);
    try {
      const results = await Promise.allSettled(secrets.map((secret) => deleteSecret(secret.secret_id)));
      const failures = results.filter((r) => r.status === 'rejected');
      if (failures.length > 0) {
        setError(
          intl.formatMessage({
            defaultMessage: 'Failed to delete some API keys. Please try again.',
            description: 'Gateway > Bulk delete API keys modal > Error message',
          }),
        );
      }
      // Always notify parent so the list refreshes (some keys may have been deleted)
      onSuccess();
      if (failures.length === 0) {
        onClose();
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClose = () => {
    setError(null);
    onClose();
  };

  return (
    <Modal
      componentId="mlflow.gateway.bulk-delete-api-key-modal"
      title={intl.formatMessage(
        {
          defaultMessage: 'Delete {count, plural, one {# API Key} other {# API Keys}}',
          description: 'Gateway > Bulk delete API keys modal > Title',
        },
        { count: secrets.length },
      )}
      visible={open}
      onCancel={handleClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.gateway.bulk-delete-api-key-modal.cancel"
            onClick={handleClose}
            disabled={isDeleting}
          >
            <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
          </Button>
          <Button
            componentId="mlflow.gateway.bulk-delete-api-key-modal.delete"
            type="primary"
            danger
            onClick={handleDelete}
            disabled={isDeleting}
            loading={isDeleting}
          >
            <FormattedMessage defaultMessage="Delete" description="Delete button text" />
          </Button>
        </div>
      }
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {error && (
          <Alert
            componentId="mlflow.gateway.bulk-delete-api-key-modal.error"
            type="error"
            message={error}
            closable={false}
          />
        )}

        <Typography.Text>
          <FormattedMessage
            defaultMessage="The following {count, plural, one {API key} other {API keys}} will be deleted:"
            description="Gateway > Bulk delete API keys modal > Description"
            values={{ count: secrets.length }}
          />
        </Typography.Text>

        <div
          css={{
            maxHeight: 200,
            overflowY: 'auto',
            border: `1px solid ${theme.colors.borderDecorative}`,
            borderRadius: theme.general.borderRadiusBase,
          }}
        >
          {secrets.map((secret) => (
            <div
              key={secret.secret_id}
              css={{
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                '&:last-child': { borderBottom: 'none' },
              }}
            >
              <Typography.Text bold>{secret.secret_name}</Typography.Text>
            </div>
          ))}
        </div>

        {hasEndpoints && (
          <Alert
            componentId="mlflow.gateway.bulk-delete-api-key-modal.warning"
            type="warning"
            message={
              <FormattedMessage
                defaultMessage="{count, plural, one {# API key is} other {# API keys are}} currently in use by endpoints. Deleting will require attaching different keys to continue using those endpoints."
                description="Gateway > Bulk delete API keys modal > Endpoints warning"
                values={{ count: secretsWithEndpoints.length }}
              />
            }
            closable={false}
          />
        )}
      </div>
    </Modal>
  );
};
