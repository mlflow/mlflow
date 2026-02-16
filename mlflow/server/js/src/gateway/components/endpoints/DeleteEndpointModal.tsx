import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { DeleteConfirmationModal } from '../common';
import { useDeleteEndpoint } from '../../hooks/useDeleteEndpoint';
import type { Endpoint, EndpointBinding, ResourceType } from '../../types';

interface DeleteEndpointModalProps {
  open: boolean;
  endpoint: Endpoint | null;
  bindings: EndpointBinding[];
  onClose: () => void;
  onSuccess?: () => void;
}

const formatResourceType = (resourceType: ResourceType): string => {
  switch (resourceType) {
    case 'scorer':
      return 'Scorer';
    default:
      return resourceType;
  }
};

export const DeleteEndpointModal = ({ open, endpoint, bindings, onClose, onSuccess }: DeleteEndpointModalProps) => {
  const { theme } = useDesignSystemTheme();
  const { mutateAsync: deleteEndpoint } = useDeleteEndpoint();

  const bindingCount = bindings.length;
  const hasBindings = bindingCount > 0;

  const handleConfirm = async () => {
    if (!endpoint) return;
    await deleteEndpoint(endpoint.endpoint_id);
    onSuccess?.();
  };

  if (!endpoint) return null;

  const renderBindingsList = () => {
    if (!hasBindings) return undefined;

    return (
      <div>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Resources using this endpoint ({count})"
            description="Gateway > Delete endpoint modal > Bindings list header"
            values={{ count: bindings.length }}
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
          {bindings.map((binding) => (
            <div
              key={`${binding.resource_type}-${binding.resource_id}`}
              css={{
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                '&:last-child': { borderBottom: 'none' },
              }}
            >
              <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>{binding.resource_id}</Typography.Text>
              <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
                {formatResourceType(binding.resource_type)}
              </Typography.Text>
            </div>
          ))}
        </div>
      </div>
    );
  };

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
            defaultMessage="This endpoint is currently in use. Deleting it will break connections to the resources listed below."
            description="Warning about resources using this endpoint"
          />
        ) : undefined
      }
      additionalContent={renderBindingsList()}
    />
  );
};
