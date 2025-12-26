import { useMemo } from 'react';
import { Drawer, Empty, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../../experiment-tracking/routes';
import type { Endpoint, EndpointBinding, ResourceType } from '../../types';

interface BindingsUsingKeyDrawerProps {
  open: boolean;
  bindings: EndpointBinding[];
  endpoints: Endpoint[];
  onClose: () => void;
}

const formatResourceType = (resourceType: ResourceType): string => {
  switch (resourceType) {
    case 'scorer_job':
      return 'Scorer Job';
    default:
      return resourceType;
  }
};

export const BindingsUsingKeyDrawer = ({ open, bindings, endpoints, onClose }: BindingsUsingKeyDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const getEndpointName = (endpointId: string) => {
    return endpoints.find((e) => e.endpoint_id === endpointId)?.name || endpointId;
  };

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
    }
  };

  const groupedBindings = useMemo(() => {
    const groups = new Map<ResourceType, EndpointBinding[]>();
    bindings.forEach((binding) => {
      const existing = groups.get(binding.resource_type) || [];
      groups.set(binding.resource_type, [...existing, binding]);
    });
    return groups;
  }, [bindings]);

  const renderResourceLink = (binding: EndpointBinding) => {
    return (
      <Typography.Text bold>
        {formatResourceType(binding.resource_type)} {binding.resource_id}
      </Typography.Text>
    );
  };

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.bindings-using-key.drawer"
        width={480}
        title={
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="Used by ({count})"
              description="Gateway > Bindings using key drawer > Title"
              values={{ count: bindings.length }}
            />
          </Typography.Title>
        }
      >
        <Spacer size="md" />
        {bindings.length === 0 ? (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No resources are using this key"
                description="Gateway > Bindings using key drawer > Empty state"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Resources using this key via endpoints"
                description="Gateway > Bindings using key drawer > Subtitle"
              />
            </Typography.Text>

            {Array.from(groupedBindings.entries()).map(([resourceType, typeBindings]) => (
              <div key={resourceType}>
                <Typography.Text bold css={{ marginBottom: theme.spacing.sm, display: 'block' }}>
                  {formatResourceType(resourceType)}s ({typeBindings.length})
                </Typography.Text>
                <div
                  css={{
                    border: `1px solid ${theme.colors.borderDecorative}`,
                    borderRadius: theme.general.borderRadiusBase,
                    overflow: 'hidden',
                  }}
                >
                  {typeBindings.map((binding, index) => (
                    <div
                      key={`${binding.endpoint_id}-${binding.resource_type}-${binding.resource_id}`}
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.xs,
                        padding: theme.spacing.sm,
                        borderBottom:
                          index < typeBindings.length - 1 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
                      }}
                    >
                      {renderResourceLink(binding)}
                      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                          <FormattedMessage
                            defaultMessage="via endpoint:"
                            description="Gateway > Bindings using key drawer > Via endpoint label"
                          />
                        </Typography.Text>
                        <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                          {getEndpointName(binding.endpoint_id)}
                        </Typography.Text>
                      </div>
                      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                        <FormattedMessage
                          defaultMessage="Bound {date}"
                          description="Gateway > Bindings using key drawer > Binding created date"
                          values={{
                            date: intl.formatDate(new Date(binding.created_at), {
                              year: 'numeric',
                              month: 'short',
                              day: 'numeric',
                            }),
                          }}
                        />
                      </Typography.Text>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};
