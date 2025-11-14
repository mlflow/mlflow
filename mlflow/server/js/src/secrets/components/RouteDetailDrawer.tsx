import {
  Button,
  Drawer,
  LightningIcon,
  PencilIcon,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { Route } from '../types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { Descriptions } from '@mlflow/mlflow/src/common/components/Descriptions';
import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';

export interface RouteDetailDrawerProps {
  route: Route | null;
  open: boolean;
  onClose: () => void;
  onUpdate?: (route: Route) => void;
  onDelete?: (route: Route) => void;
}

const ProviderBadge = ({ provider }: { provider?: string }) => {
  const { theme } = useDesignSystemTheme();

  const getProviderColor = (provider?: string) => {
    if (!provider) {
      return { bg: theme.colors.backgroundSecondary, text: theme.colors.textSecondary };
    }
    switch (provider.toLowerCase()) {
      case 'openai':
        return { bg: '#10A37F15', text: '#10A37F' };
      case 'anthropic':
        return { bg: '#D4915215', text: '#D49152' };
      case 'bedrock':
      case 'aws':
        return { bg: '#FF990015', text: '#FF9900' };
      case 'vertex_ai':
      case 'google':
        return { bg: '#4285F415', text: '#4285F4' };
      case 'azure':
        return { bg: '#0078D415', text: '#0078D4' };
      case 'databricks':
        return { bg: '#FF362115', text: '#FF3621' };
      default:
        return { bg: theme.colors.backgroundSecondary, text: theme.colors.textSecondary };
    }
  };

  const colors = getProviderColor(provider);

  return (
    <div
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '4px 8px',
        gap: theme.spacing.xs,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: colors.bg,
        color: colors.text,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: 600,
      }}
    >
      <LightningIcon css={{ fontSize: 12 }} />
      {provider || 'Unknown'}
    </div>
  );
};

export const RouteDetailDrawer = ({ route, open, onClose, onUpdate, onDelete }: RouteDetailDrawerProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  // Convert tags to array format
  const tagEntities = route?.tags
    ? Array.isArray(route.tags)
      ? route.tags
      : Object.entries(route.tags).map(([key, value]) => ({ key, value }))
    : [];

  return (
    <Drawer.Root modal open={open} onOpenChange={onClose}>
      <Drawer.Content
        componentId="mlflow.routes.detail_drawer"
        width="700px"
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 32,
                height: 32,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundSecondary,
              }}
            >
              <LightningIcon css={{ fontSize: 18 }} />
            </div>
            <FormattedMessage defaultMessage="Route Details" description="Route detail drawer > drawer title" />
          </div>
        }
      >
        {route && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            {/* Header section with name, provider, and model */}
            <div
              css={{
                padding: theme.spacing.md,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundSecondary,
                border: `1px solid ${theme.colors.border}`,
              }}
            >
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: theme.spacing.sm,
                }}
              >
                <Typography.Title level={3} css={{ margin: 0 }}>
                  {route.name || route.route_id}
                </Typography.Title>
                <ProviderBadge provider={route.provider} />
              </div>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <Typography.Text color="secondary" size="sm">
                    <FormattedMessage defaultMessage="Model:" description="Route detail drawer > model label" />
                  </Typography.Text>
                  <Tag componentId="mlflow.routes.detail_drawer.model_tag">
                    <Typography.Text>{route.model_name}</Typography.Text>
                  </Tag>
                </div>

                {route.description && (
                  <div>
                    <Typography.Text color="secondary" size="sm">
                      <FormattedMessage
                        defaultMessage="Description:"
                        description="Route detail drawer > description label"
                      />
                    </Typography.Text>
                    <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0 }}>
                      {route.description}
                    </Typography.Paragraph>
                  </div>
                )}
              </div>
            </div>

            {/* Metadata section */}
            <div>
              <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Configuration"
                  description="Route detail drawer > configuration section title"
                />
              </Typography.Title>
              <Descriptions columns={1}>
                <Descriptions.Item
                  label={
                    <FormattedMessage
                      defaultMessage="Secret Name"
                      description="Route detail drawer > secret name label"
                    />
                  }
                >
                  <Typography.Text css={{ fontWeight: 500 }}>{route.secret_name || route.secret_id}</Typography.Text>
                </Descriptions.Item>
                {route.created_by && (
                  <Descriptions.Item
                    label={
                      <FormattedMessage
                        defaultMessage="Created By"
                        description="Route detail drawer > created by label"
                      />
                    }
                  >
                    <Typography.Text>{route.created_by}</Typography.Text>
                  </Descriptions.Item>
                )}
                <Descriptions.Item
                  label={
                    <FormattedMessage
                      defaultMessage="Created At"
                      description="Route detail drawer > created at label"
                    />
                  }
                >
                  <Typography.Text>{Utils.formatTimestamp(route.created_at)}</Typography.Text>
                </Descriptions.Item>
                {route.last_updated_by && (
                  <Descriptions.Item
                    label={
                      <FormattedMessage
                        defaultMessage="Last Updated By"
                        description="Route detail drawer > last updated by label"
                      />
                    }
                  >
                    <Typography.Text>{route.last_updated_by}</Typography.Text>
                  </Descriptions.Item>
                )}
                <Descriptions.Item
                  label={
                    <FormattedMessage
                      defaultMessage="Last Updated"
                      description="Route detail drawer > last updated label"
                    />
                  }
                >
                  <Typography.Text>{Utils.formatTimestamp(route.last_updated_at)}</Typography.Text>
                </Descriptions.Item>
              </Descriptions>
            </div>

            {/* Tags section */}
            {tagEntities.length > 0 && (
              <div>
                <Typography.Title level={4} css={{ marginBottom: theme.spacing.md }}>
                  <FormattedMessage defaultMessage="Tags" description="Route detail drawer > tags section title" />
                </Typography.Title>
                <div css={{ display: 'flex', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
                  {tagEntities.map((tag) => (
                    <KeyValueTag key={tag.key} tag={tag} />
                  ))}
                </div>
              </div>
            )}

            {/* Action buttons */}
            <div css={{ display: 'flex', gap: theme.spacing.sm }}>
              <Button
                componentId="mlflow.routes.detail_drawer.update_button"
                icon={<PencilIcon />}
                onClick={() => {
                  onUpdate?.(route);
                }}
              >
                <FormattedMessage defaultMessage="Update Route" description="Route detail drawer > update button" />
              </Button>
              <Button
                componentId="mlflow.routes.detail_drawer.delete_button"
                icon={<TrashIcon />}
                danger
                onClick={() => {
                  onDelete?.(route);
                }}
              >
                <FormattedMessage defaultMessage="Delete Route" description="Route detail drawer > delete button" />
              </Button>
            </div>
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};
