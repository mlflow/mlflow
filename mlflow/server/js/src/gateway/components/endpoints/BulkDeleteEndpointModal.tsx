import { useState } from 'react';
import { Alert, Button, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDeleteEndpoint } from '../../hooks/useDeleteEndpoint';
import type { Endpoint, EndpointBinding } from '../../types';

interface BulkDeleteEndpointModalProps {
  open: boolean;
  endpoints: Endpoint[];
  getBindingsForEndpoint: (endpointId: string) => EndpointBinding[];
  onClose: () => void;
  onSuccess: () => void;
}

export const BulkDeleteEndpointModal = ({
  open,
  endpoints,
  getBindingsForEndpoint,
  onClose,
  onSuccess,
}: BulkDeleteEndpointModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { mutateAsync: deleteEndpoint } = useDeleteEndpoint();
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const endpointsWithBindings = endpoints.filter((ep) => getBindingsForEndpoint(ep.endpoint_id).length > 0);
  const hasBindings = endpointsWithBindings.length > 0;

  const handleDelete = async () => {
    setIsDeleting(true);
    setError(null);
    try {
      await Promise.all(
        endpoints.map((endpoint) =>
          deleteEndpoint({ endpointId: endpoint.endpoint_id, modelMappings: endpoint.model_mappings }),
        ),
      );
      onSuccess();
      onClose();
    } catch {
      setError(
        intl.formatMessage({
          defaultMessage: 'Failed to delete some endpoints. Please try again.',
          description: 'Gateway > Bulk delete endpoints modal > Error message',
        }),
      );
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
      componentId="mlflow.gateway.delete-endpoint-modal"
      title={intl.formatMessage(
        {
          defaultMessage: 'Delete {count, plural, one {# Endpoint} other {# Endpoints}}',
          description: 'Gateway > Bulk delete endpoints modal > Title',
        },
        { count: endpoints.length },
      )}
      visible={open}
      onCancel={handleClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.gateway.delete-endpoint-modal.cancel" onClick={handleClose} disabled={isDeleting}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
          </Button>
          <Button
            componentId="mlflow.gateway.delete-endpoint-modal.delete"
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
            componentId="mlflow.gateway.delete-endpoint-modal.error"
            type="error"
            message={error}
            closable={false}
          />
        )}

        <Typography.Text>
          <FormattedMessage
            defaultMessage="The following {count, plural, one {endpoint} other {endpoints}} will be deleted:"
            description="Gateway > Bulk delete endpoints modal > Description"
            values={{ count: endpoints.length }}
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
          {endpoints.map((endpoint) => (
            <div
              key={endpoint.endpoint_id}
              css={{
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                '&:last-child': { borderBottom: 'none' },
              }}
            >
              <Typography.Text bold>{endpoint.name ?? endpoint.endpoint_id}</Typography.Text>
            </div>
          ))}
        </div>

        {hasBindings && (
          <Alert
            componentId="mlflow.gateway.delete-endpoint-modal.warning"
            type="warning"
            message={
              <FormattedMessage
                defaultMessage="{count, plural, one {# endpoint is} other {# endpoints are}} currently in use. Deleting will break connections to resources using them."
                description="Gateway > Bulk delete endpoints modal > Bindings warning"
                values={{ count: endpointsWithBindings.length }}
              />
            }
            closable={false}
          />
        )}
      </div>
    </Modal>
  );
};
