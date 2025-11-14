import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Drawer,
  FormUI,
  Input,
  LightningIcon,
  PencilIcon,
  RefreshIcon,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState, useMemo, useEffect } from 'react';
// eslint-disable-next-line import/no-extraneous-dependencies
import { notification } from 'antd';
import type { Route } from '../types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { Descriptions } from '@mlflow/mlflow/src/common/components/Descriptions';
import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';
import { MaskedApiKeyInput } from './MaskedApiKeyInput';
import { ProviderBadge } from './ProviderBadge';
import { useListSecrets } from '../hooks/useListSecrets';
import { useListBindings } from '../hooks/useListBindings';
import { useListRoutes } from '../hooks/useListRoutes';
import { useUpdateSecretMutation } from '../hooks/useUpdateSecretMutation';
import { useDeleteSecretMutation } from '../hooks/useDeleteSecretMutation';

export interface RouteDetailDrawerProps {
  route: Route | null;
  open: boolean;
  onClose: () => void;
  onUpdate?: (route: Route) => void;
  onDelete?: (route: Route) => void;
}

export const RouteDetailDrawer = ({ route, open, onClose, onUpdate, onDelete }: RouteDetailDrawerProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  // Route management state
  const [routeManagementExpanded, setRouteManagementExpanded] = useState(false);
  const [routeDeleteConfirmation, setRouteDeleteConfirmation] = useState('');

  // Secret management state
  const [secretManagementExpanded, setSecretManagementExpanded] = useState(false);
  const [managementMode, setManagementMode] = useState<'view' | 'rotate'>('view');
  const [newSecretValue, setNewSecretValue] = useState('');
  const [secretDeleteConfirmation, setSecretDeleteConfirmation] = useState('');
  const [error, setError] = useState<string | undefined>(undefined);

  // Fetch data for secret management
  const { secrets = [] } = useListSecrets({ enabled: open });
  const { routes = [] } = useListRoutes({ enabled: open });
  const { bindings = [] } = useListBindings({
    secretId: route?.secret_id || '',
    enabled: open && !!route?.secret_id,
  });

  // Get the current secret
  const currentSecret = useMemo(() => {
    return secrets.find((s) => s.secret_id === route?.secret_id);
  }, [secrets, route?.secret_id]);

  // Calculate impact - routes using this secret and their bindings
  const routesUsingSecret = useMemo(() => {
    if (!currentSecret) return [];
    return routes.filter((r) => r.secret_id === currentSecret.secret_id);
  }, [routes, currentSecret]);

  const resourceBindings = useMemo(() => {
    if (!currentSecret) return [];
    // Filter out GLOBAL bindings - only show specific resource bindings
    return bindings.filter((b) => b.secret_id === currentSecret.secret_id && b.resource_type !== 'GLOBAL');
  }, [bindings, currentSecret]);

  // Reset management sections when route changes
  useEffect(() => {
    setRouteManagementExpanded(false);
    setRouteDeleteConfirmation('');
    setSecretManagementExpanded(false);
    setManagementMode('view');
    setNewSecretValue('');
    setSecretDeleteConfirmation('');
    setError(undefined);
  }, [route?.route_id]);

  // Mutation hooks
  const { updateSecret, isLoading: isUpdating } = useUpdateSecretMutation({
    onSuccess: () => {
      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Key rotated successfully',
          description: 'Route detail drawer > rotate key success',
        }),
      });
      setManagementMode('view');
      setNewSecretValue('');
      setError(undefined);
    },
    onError: (error) => {
      setError(error.message);
    },
  });

  const { deleteSecret, isLoading: isDeleting } = useDeleteSecretMutation({
    onSuccess: () => {
      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Key deleted successfully',
          description: 'Route detail drawer > delete key success',
        }),
      });
      onClose(); // Close the drawer since the route was cascade-deleted
    },
    onError: (error) => {
      setError(error.message);
    },
  });

  const handleRotateKey = () => {
    if (!currentSecret || !newSecretValue.trim()) {
      setError(
        intl.formatMessage({
          defaultMessage: 'New API key is required',
          description: 'Route detail drawer > rotate key validation error',
        }),
      );
      return;
    }
    setError(undefined);
    updateSecret({
      secret_id: currentSecret.secret_id,
      secret_value: newSecretValue,
    });
  };

  const handleDeleteKey = () => {
    if (!currentSecret) return;
    setError(undefined);
    deleteSecret({
      secret_id: currentSecret.secret_id,
    });
  };

  const handleCancelManagement = () => {
    setManagementMode('view');
    setNewSecretValue('');
    setSecretDeleteConfirmation('');
    setError(undefined);
  };

  const handleDeleteRoute = () => {
    if (!route) return;
    onDelete?.(route);
    setRouteDeleteConfirmation('');
  };

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

                  {tagEntities.length > 0 && (
                    <div>
                      <Typography.Text color="secondary" size="sm">
                        <FormattedMessage defaultMessage="Tags:" description="Route detail drawer > tags label" />
                      </Typography.Text>
                      <div css={{ display: 'flex', gap: theme.spacing.sm, flexWrap: 'wrap', marginTop: theme.spacing.xs }}>
                        {tagEntities.map((tag) => (
                          <KeyValueTag key={tag.key} tag={tag} />
                        ))}
                      </div>
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

              {/* Management Operations Header */}
              <div>
                <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.xs }}>
                  <FormattedMessage
                    defaultMessage="Management"
                    description="Route detail drawer > management operations section title"
                  />
                </Typography.Title>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="Update the route configuration, manage the API key, or remove the route."
                    description="Route detail drawer > management operations description"
                  />
                </Typography.Text>
              </div>

              {/* Route Management section */}
              <div>
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.sm,
                    cursor: 'pointer',
                    padding: theme.spacing.sm,
                    marginLeft: -theme.spacing.sm,
                    marginRight: -theme.spacing.sm,
                    borderRadius: theme.borders.borderRadiusMd,
                    '&:hover': {
                      backgroundColor: theme.colors.backgroundSecondary,
                    },
                  }}
                  onClick={() => {
                    setRouteManagementExpanded(!routeManagementExpanded);
                    if (!routeManagementExpanded) {
                      setRouteDeleteConfirmation('');
                    }
                  }}
                >
                  {routeManagementExpanded ? (
                    <ChevronDownIcon css={{ fontSize: 16 }} />
                  ) : (
                    <ChevronRightIcon css={{ fontSize: 16 }} />
                  )}
                  <Typography.Title level={4} css={{ margin: 0 }}>
                    <FormattedMessage
                      defaultMessage="Route Management"
                      description="Route detail drawer > route management section title"
                    />
                  </Typography.Title>
                </div>

                {routeManagementExpanded && route && (
                  <div
                    css={{
                      marginTop: theme.spacing.md,
                      padding: theme.spacing.md,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `1px solid ${theme.colors.border}`,
                      backgroundColor: theme.colors.backgroundSecondary,
                    }}
                  >
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                      <div>
                        <Button
                          componentId="mlflow.routes.detail_drawer.update_route_button"
                          icon={<PencilIcon />}
                          onClick={() => {
                            onUpdate?.(route);
                          }}
                        >
                          <FormattedMessage
                            defaultMessage="Update Route"
                            description="Route detail drawer > update route button"
                          />
                        </Button>
                      </div>

                      <div
                        css={{
                          padding: theme.spacing.md,
                          borderRadius: theme.borders.borderRadiusMd,
                          backgroundColor: theme.colors.backgroundDanger,
                          border: `1px solid ${theme.colors.borderDanger}`,
                        }}
                      >
                        <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.sm, fontWeight: 600 }}>
                          <FormattedMessage
                            defaultMessage="Delete Confirmation Required"
                            description="Route detail drawer > delete confirmation title"
                          />
                        </Typography.Text>
                        <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md }} size="sm">
                          <FormattedMessage
                            defaultMessage="To delete this route, type its name below to confirm:"
                            description="Route detail drawer > delete confirmation description"
                          />
                        </Typography.Text>
                        <div>
                          <FormUI.Label htmlFor="route-delete-confirmation">
                            <FormattedMessage
                              defaultMessage="Route Name"
                              description="Route detail drawer > route name label"
                            />
                            <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                          </FormUI.Label>
                          <Input
                            id="route-delete-confirmation"
                            componentId="mlflow.routes.detail_drawer.delete_confirmation_input"
                            value={routeDeleteConfirmation}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setRouteDeleteConfirmation(e.target.value)}
                            placeholder={route.name || route.route_id}
                          />
                          <Typography.Text
                            size="sm"
                            color="secondary"
                            css={{ display: 'block', marginTop: theme.spacing.xs }}
                          >
                            <FormattedMessage
                              defaultMessage='Enter "{name}" to enable deletion'
                              description="Route detail drawer > delete confirmation help"
                              values={{ name: route.name || route.route_id }}
                            />
                          </Typography.Text>
                          <Button
                            componentId="mlflow.routes.detail_drawer.delete_route_button"
                            icon={<TrashIcon />}
                            danger
                            onClick={handleDeleteRoute}
                            disabled={routeDeleteConfirmation !== (route.name || route.route_id)}
                            css={{ marginTop: theme.spacing.md }}
                          >
                            <FormattedMessage
                              defaultMessage="Delete Route"
                              description="Route detail drawer > delete route button"
                            />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Secret Management section */}
              <div>
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.sm,
                    cursor: 'pointer',
                    padding: theme.spacing.sm,
                    marginLeft: -theme.spacing.sm,
                    marginRight: -theme.spacing.sm,
                    borderRadius: theme.borders.borderRadiusMd,
                    '&:hover': {
                      backgroundColor: theme.colors.backgroundSecondary,
                    },
                  }}
                  onClick={() => {
                    setSecretManagementExpanded(!secretManagementExpanded);
                    if (!secretManagementExpanded) {
                      setManagementMode('view');
                      setNewSecretValue('');
                      setError(undefined);
                    }
                  }}
                >
                  {secretManagementExpanded ? (
                    <ChevronDownIcon css={{ fontSize: 16 }} />
                  ) : (
                    <ChevronRightIcon css={{ fontSize: 16 }} />
                  )}
                  <Typography.Title level={4} css={{ margin: 0 }}>
                    <FormattedMessage
                      defaultMessage="Secret Management"
                      description="Route detail drawer > secret management section title"
                    />
                  </Typography.Title>
                </div>

                {secretManagementExpanded && currentSecret && (
                  <div
                    css={{
                      marginTop: theme.spacing.md,
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: theme.spacing.md,
                    }}
                  >
                      {/* Left panel: Management UI */}
                    <div
                      css={{
                        padding: theme.spacing.md,
                        borderRadius: theme.borders.borderRadiusMd,
                        border: `1px solid ${theme.colors.border}`,
                        backgroundColor: theme.colors.backgroundSecondary,
                      }}
                    >
                      {managementMode === 'view' && (
                      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                          <div>
                            <Typography.Text color="secondary" size="sm">
                              <FormattedMessage
                                defaultMessage="Secret Name"
                                description="Route detail drawer > secret name label"
                              />
                            </Typography.Text>
                            <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontWeight: 500 }}>
                              {currentSecret.secret_name}
                            </Typography.Paragraph>
                          </div>
                          <div>
                            <Typography.Text color="secondary" size="sm">
                              <FormattedMessage
                                defaultMessage="Masked Value"
                                description="Route detail drawer > masked value label"
                              />
                            </Typography.Text>
                            <Typography.Paragraph
                              css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontFamily: 'monospace' }}
                            >
                              {currentSecret.masked_value}
                            </Typography.Paragraph>
                          </div>
                          {currentSecret.provider && (
                            <div>
                              <Typography.Text color="secondary" size="sm">
                                <FormattedMessage
                                  defaultMessage="Provider"
                                  description="Route detail drawer > provider label"
                                />
                              </Typography.Text>
                              <div css={{ marginTop: theme.spacing.xs }}>
                                <ProviderBadge provider={currentSecret.provider} />
                              </div>
                            </div>
                          )}
                        </div>
                        <div>
                          <Button
                            componentId="mlflow.routes.detail_drawer.rotate_key_button"
                            icon={<RefreshIcon />}
                            onClick={() => setManagementMode('rotate')}
                          >
                            <FormattedMessage
                              defaultMessage="Rotate Key"
                              description="Route detail drawer > rotate button"
                            />
                          </Button>
                          <Typography.Text
                            color="secondary"
                            size="sm"
                            css={{ display: 'block', marginTop: theme.spacing.xs }}
                          >
                            <FormattedMessage
                              defaultMessage="Replace the API key value while keeping the same route configuration."
                              description="Route detail drawer > rotate key explanation"
                            />
                          </Typography.Text>
                        </div>

                        <div
                          css={{
                            padding: theme.spacing.md,
                            borderRadius: theme.borders.borderRadiusMd,
                            backgroundColor: theme.colors.backgroundDanger,
                            border: `1px solid ${theme.colors.borderDanger}`,
                          }}
                        >
                          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.sm, fontWeight: 600 }}>
                            <FormattedMessage
                              defaultMessage="Delete Confirmation Required"
                              description="Route detail drawer > delete key confirmation title"
                            />
                          </Typography.Text>
                          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md }} size="sm">
                            <FormattedMessage
                              defaultMessage="This action cannot be undone and will cascade delete all routes and resource bindings using this key. Type the secret name below to confirm:"
                              description="Route detail drawer > delete key confirmation description"
                            />
                          </Typography.Text>
                          <div>
                            <FormUI.Label htmlFor="secret-delete-confirmation">
                              <FormattedMessage
                                defaultMessage="Secret Name"
                                description="Route detail drawer > secret name confirmation label"
                              />
                              <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                            </FormUI.Label>
                            <Input
                              id="secret-delete-confirmation"
                              componentId="mlflow.routes.detail_drawer.secret_delete_confirmation_input"
                              value={secretDeleteConfirmation}
                              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSecretDeleteConfirmation(e.target.value)}
                              placeholder={currentSecret?.secret_name || ''}
                            />
                            <Typography.Text
                              size="sm"
                              color="secondary"
                              css={{ display: 'block', marginTop: theme.spacing.xs }}
                            >
                              <FormattedMessage
                                defaultMessage='Enter "{name}" to enable deletion'
                                description="Route detail drawer > secret delete confirmation help"
                                values={{ name: currentSecret?.secret_name || '' }}
                              />
                            </Typography.Text>
                            <Button
                              componentId="mlflow.routes.detail_drawer.delete_key_button"
                              icon={<TrashIcon />}
                              danger
                              onClick={handleDeleteKey}
                              disabled={secretDeleteConfirmation !== currentSecret?.secret_name}
                              css={{ marginTop: theme.spacing.md }}
                            >
                              <FormattedMessage
                                defaultMessage="Delete Key"
                                description="Route detail drawer > delete key button"
                              />
                            </Button>
                          </div>
                        </div>
                      </div>
                    )}

                    {managementMode === 'rotate' && (
                      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                        <div>
                          <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.sm }}>
                            <FormattedMessage
                              defaultMessage="Rotate Key"
                              description="Route detail drawer > rotate key title"
                            />
                          </Typography.Title>
                          <Typography.Text color="secondary" size="sm">
                            <FormattedMessage
                              defaultMessage="Enter a new API key value. The route will automatically use the new value."
                              description="Route detail drawer > rotate key description"
                            />
                          </Typography.Text>
                        </div>

                        <div>
                          <FormUI.Label htmlFor="new-secret-value">
                            <FormattedMessage
                              defaultMessage="New API Key"
                              description="Route detail drawer > new key label"
                            />
                            <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                          </FormUI.Label>
                          <MaskedApiKeyInput
                            value={newSecretValue}
                            onChange={setNewSecretValue}
                            placeholder={intl.formatMessage({
                              defaultMessage: 'Enter new API key',
                              description: 'Route detail drawer > new key placeholder',
                            })}
                            id="new-secret-value"
                            componentId="mlflow.routes.detail_drawer.new_secret_value"
                          />
                          {error && <FormUI.Message type="error" message={error} />}
                        </div>

                        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
                          <Button
                            componentId="mlflow.routes.detail_drawer.confirm_rotate_button"
                            onClick={handleRotateKey}
                            loading={isUpdating}
                            disabled={!newSecretValue.trim()}
                          >
                            <FormattedMessage
                              defaultMessage="Rotate Key"
                              description="Route detail drawer > confirm rotate button"
                            />
                          </Button>
                          <Button
                            componentId="mlflow.routes.detail_drawer.cancel_rotate_button"
                            onClick={handleCancelManagement}
                            disabled={isUpdating}
                          >
                            <FormattedMessage
                              defaultMessage="Cancel"
                              description="Route detail drawer > cancel button"
                            />
                          </Button>
                        </div>
                      </div>
                    )}
                    </div>

                    {/* Right panel: Impact Analysis */}
                    <div
                      css={{
                        padding: theme.spacing.md,
                        borderRadius: theme.borders.borderRadiusMd,
                        border: `1px solid ${theme.colors.border}`,
                        backgroundColor: theme.colors.backgroundSecondary,
                      }}
                    >
                      <Typography.Title level={4} css={{ marginTop: 0, marginBottom: theme.spacing.md }}>
                        <FormattedMessage
                          defaultMessage="Impact Analysis"
                          description="Route detail drawer > impact analysis title"
                        />
                      </Typography.Title>

                      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                        {/* Summary */}
                        <div
                          css={{
                            padding: theme.spacing.sm,
                            borderRadius: theme.borders.borderRadiusMd,
                            backgroundColor: theme.colors.backgroundWarning,
                            border: `1px solid ${theme.colors.borderWarning}`,
                          }}
                        >
                          <Typography.Text css={{ display: 'block', fontWeight: 600 }} size="sm">
                            <FormattedMessage
                              defaultMessage="This key is used by:"
                              description="Route detail drawer > impact analysis header"
                            />
                          </Typography.Text>
                          <Typography.Text css={{ display: 'block', marginTop: theme.spacing.xs }} size="sm">
                            <FormattedMessage
                              defaultMessage="{routeCount} {routeCount, plural, one {route} other {routes}}, {bindingCount} {bindingCount, plural, one {resource binding} other {resource bindings}}"
                              description="Route detail drawer > impact summary"
                              values={{
                                routeCount: routesUsingSecret.length,
                                bindingCount: resourceBindings.length,
                              }}
                            />
                          </Typography.Text>
                        </div>

                        {/* Routes List */}
                        {routesUsingSecret.length > 0 && (
                          <div>
                            <Typography.Text
                              css={{ display: 'block', marginBottom: theme.spacing.sm, fontWeight: 600 }}
                              size="sm"
                            >
                              <FormattedMessage
                                defaultMessage="Affected Routes"
                                description="Route detail drawer > affected routes title"
                              />
                            </Typography.Text>
                            <div
                              css={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: theme.spacing.sm,
                                maxHeight: 300,
                                overflowY: 'auto',
                              }}
                            >
                              {routesUsingSecret.map((affectedRoute) => {
                                // Get resource bindings for this route, excluding GLOBAL bindings and self-bindings
                                const routeBindings = bindings.filter(
                                  (b) =>
                                    b.route_id === affectedRoute.route_id &&
                                    b.resource_type !== 'GLOBAL' &&
                                    !(
                                      b.resource_type?.toLowerCase() === 'route' &&
                                      b.resource_id === affectedRoute.route_id
                                    ),
                                );
                                return (
                                  <div
                                    key={affectedRoute.route_id}
                                    css={{
                                      padding: theme.spacing.sm,
                                      borderRadius: theme.borders.borderRadiusMd,
                                      border: `1px solid ${theme.colors.border}`,
                                      backgroundColor: theme.colors.backgroundPrimary,
                                    }}
                                  >
                                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBottom: theme.spacing.xs }}>
                                      {affectedRoute.route_id === route?.route_id && (
                                        <Tag componentId="mlflow.routes.detail_drawer.current_route_tag">
                                          <Typography.Text size="sm">
                                            <FormattedMessage
                                              defaultMessage="Current"
                                              description="Route detail drawer > current route badge"
                                            />
                                          </Typography.Text>
                                        </Tag>
                                      )}
                                      <Typography.Text css={{ fontWeight: 600 }} size="sm">
                                        {affectedRoute.name || affectedRoute.route_id}
                                      </Typography.Text>
                                    </div>
                                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBottom: theme.spacing.xs }}>
                                      <Typography.Text color="secondary" size="sm">
                                        <FormattedMessage
                                          defaultMessage="Model:"
                                          description="Route detail drawer > model label"
                                        />
                                      </Typography.Text>
                                      <Typography.Text size="sm">{affectedRoute.model_name}</Typography.Text>
                                    </div>
                                    {routeBindings.length > 0 && (
                                      <div>
                                        <Typography.Text
                                          color="secondary"
                                          size="sm"
                                          css={{ display: 'block', marginBottom: theme.spacing.xs }}
                                        >
                                          <FormattedMessage
                                            defaultMessage="Resource Bindings ({count}):"
                                            description="Route detail drawer > resource bindings label"
                                            values={{ count: routeBindings.length }}
                                          />
                                        </Typography.Text>
                                        <div css={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                                          {routeBindings.slice(0, 3).map((binding) => (
                                            <Typography.Text
                                              key={binding.binding_id}
                                              size="sm"
                                              css={{
                                                fontFamily: 'monospace',
                                                fontSize: theme.typography.fontSizeSm,
                                                color: theme.colors.textSecondary,
                                              }}
                                            >
                                              {binding.resource_type}: {binding.resource_id}
                                            </Typography.Text>
                                          ))}
                                          {routeBindings.length > 3 && (
                                            <Typography.Text size="sm" color="secondary">
                                              <FormattedMessage
                                                defaultMessage="+ {count} more"
                                                description="Route detail drawer > more bindings"
                                                values={{ count: routeBindings.length - 3 }}
                                              />
                                            </Typography.Text>
                                          )}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Standalone Resource Bindings (not route bindings) */}
                        {resourceBindings.filter((b) => !b.route_id).length > 0 && (
                          <div>
                            <Typography.Text
                              css={{ display: 'block', marginBottom: theme.spacing.sm, fontWeight: 600 }}
                              size="sm"
                            >
                              <FormattedMessage
                                defaultMessage="Direct Resource Bindings"
                                description="Route detail drawer > direct bindings title"
                              />
                            </Typography.Text>
                            <div
                              css={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 4,
                                maxHeight: 150,
                                overflowY: 'auto',
                              }}
                            >
                              {resourceBindings
                                .filter((b) => !b.route_id)
                                .slice(0, 5)
                                .map((binding) => (
                                  <Typography.Text
                                    key={binding.binding_id}
                                    size="sm"
                                    css={{
                                      fontFamily: 'monospace',
                                      fontSize: theme.typography.fontSizeSm,
                                      color: theme.colors.textSecondary,
                                    }}
                                  >
                                    {binding.resource_type}: {binding.resource_id}
                                  </Typography.Text>
                                ))}
                              {resourceBindings.filter((b) => !b.route_id).length > 5 && (
                                <Typography.Text size="sm" color="secondary">
                                  <FormattedMessage
                                    defaultMessage="+ {count} more"
                                    description="Route detail drawer > more bindings"
                                    values={{ count: resourceBindings.filter((b) => !b.route_id).length - 5 }}
                                  />
                                </Typography.Text>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
      </Drawer.Content>
    </Drawer.Root>
  );
};
