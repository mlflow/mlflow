import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Drawer,
  Empty,
  GearIcon,
  PencilIcon,
  Spinner,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Secret } from '../types';
import { useListSecrets } from '../hooks/useListSecrets';
import { useListBindings } from '../hooks/useListBindings';
import { useListEndpoints } from '../hooks/useListEndpoints';
import { DeleteSecretModal } from './DeleteSecretModal';
import { UpdateSecretModal } from './UpdateSecretModal';

export interface SecretManagementDrawerProps {
  open: boolean;
  onClose: () => void;
}

interface ExpandedRowData {
  bindings: any[];
  isLoading: boolean;
}

export const SecretManagementDrawer = ({ open, onClose }: SecretManagementDrawerProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { secrets = [], isLoading: isLoadingSecrets } = useListSecrets({ enabled: open });
  const { endpoints = [] } = useListEndpoints({ enabled: open });
  const [expandedSecretIds, setExpandedSecretIds] = useState<Set<string>>(new Set());
  const [deleteSecret, setDeleteSecret] = useState<Secret | null>(null);
  const [updateSecret, setUpdateSecret] = useState<Secret | null>(null);

  // Auto-collapse all expanded secrets when drawer closes
  useEffect(() => {
    if (!open) {
      setExpandedSecretIds(new Set());
    }
  }, [open]);

  const toggleExpanded = useCallback((secretId: string) => {
    setExpandedSecretIds((prev) => {
      const next = new Set(prev);
      if (next.has(secretId)) {
        next.delete(secretId);
      } else {
        next.add(secretId);
      }
      return next;
    });
  }, []);

  const handleDeleteClick = useCallback(
    (secret: Secret) => {
      setDeleteSecret(secret);
      onClose();
    },
    [onClose],
  );

  const handleUpdateClick = useCallback(
    (secret: Secret) => {
      setUpdateSecret(secret);
      onClose();
    },
    [onClose],
  );

  // Create a map of secret_id -> endpoint names for display
  const secretToEndpoints = useMemo(() => {
    const map = new Map<string, string[]>();
    endpoints.forEach((endpoint) => {
      if (endpoint.secret_id) {
        const existing = map.get(endpoint.secret_id) || [];
        map.set(endpoint.secret_id, [...existing, endpoint.name || endpoint.endpoint_id]);
      }
    });
    return map;
  }, [endpoints]);

  if (isLoadingSecrets) {
    return (
      <Drawer.Root modal open={open} onOpenChange={onClose}>
        <Drawer.Content
          componentId="mlflow.secrets.management_drawer"
          width="1000px"
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
                <GearIcon css={{ fontSize: 18 }} />
              </div>
              <FormattedMessage defaultMessage="Manage Secrets" description="Secret management drawer > drawer title" />
            </div>
          }
        >
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
            <Spinner />
          </div>
        </Drawer.Content>
      </Drawer.Root>
    );
  }

  return (
    <>
      <Drawer.Root modal open={open} onOpenChange={onClose}>
        <Drawer.Content
          componentId="mlflow.secrets.management_drawer"
          width="1000px"
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
                <GearIcon css={{ fontSize: 18 }} />
              </div>
              <FormattedMessage defaultMessage="Manage Secrets" description="Secret management drawer > drawer title" />
            </div>
          }
        >
          {secrets.length === 0 ? (
            <Empty
              description={
                <FormattedMessage
                  defaultMessage="No secrets found"
                  description="Secret management drawer > no secrets"
                />
              }
            />
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column' }}>
              <div
                css={{
                  padding: theme.spacing.md,
                  marginBottom: theme.spacing.md,
                  backgroundColor: theme.colors.backgroundSecondary,
                  borderRadius: theme.borders.borderRadiusMd,
                  border: `1px solid ${theme.colors.border}`,
                }}
              >
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="View and manage all secrets. Click on any secret to expand details. Secrets with 0 usage can be safely deleted."
                    description="Secret management drawer > description"
                  />
                </Typography.Text>
              </div>

              <div css={{ display: 'flex', flexDirection: 'column' }}>
                {secrets.map((secret) => (
                  <SecretManagementRow
                    key={secret.secret_id}
                    secret={secret}
                    isExpanded={expandedSecretIds.has(secret.secret_id)}
                    onToggleExpand={() => toggleExpanded(secret.secret_id)}
                    onUpdate={() => handleUpdateClick(secret)}
                    onDelete={() => handleDeleteClick(secret)}
                    routeNames={secretToEndpoints.get(secret.secret_id) || []}
                  />
                ))}
              </div>
            </div>
          )}
        </Drawer.Content>
      </Drawer.Root>

      <UpdateSecretModal secret={updateSecret} visible={!!updateSecret} onCancel={() => setUpdateSecret(null)} />
      <DeleteSecretModal secret={deleteSecret} visible={!!deleteSecret} onCancel={() => setDeleteSecret(null)} />
    </>
  );
};

interface SecretManagementRowProps {
  secret: Secret;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onUpdate: () => void;
  onDelete: () => void;
  routeNames: string[];
}

const SecretManagementRow = ({
  secret,
  isExpanded,
  onToggleExpand,
  onUpdate,
  onDelete,
  routeNames,
}: SecretManagementRowProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { bindings = [], isLoading: isLoadingBindings } = useListBindings({
    secretId: secret.secret_id,
    enabled: isExpanded,
  });

  // Use the binding_count from the secret object for display, not bindings.length
  const usageCount = secret.binding_count ?? 0;
  const hasZeroUsage = usageCount === 0;

  const formatResourceType = (binding: any) => {
    // If this is a route binding, show "Route" regardless of the resource_type field
    if (binding.endpoint_id) {
      return 'Route';
    }
    // Convert SCORER_JOB to "Scorer Job", GLOBAL to "Global", etc.
    return binding.resource_type
      .split('_')
      .map((word: string) => word.charAt(0) + word.slice(1).toLowerCase())
      .join(' ');
  };

  const getResourceDisplay = (binding: any) => {
    // For route bindings, show the route name instead of resource_id
    if (binding.endpoint_id) {
      return binding.route_name || binding.endpoint_id;
    }
    return binding.resource_id;
  };

  return (
    <div
      css={{
        marginBottom: theme.spacing.sm,
        border: `2px solid ${hasZeroUsage ? theme.colors.borderValidationDanger : theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundPrimary,
        overflow: 'hidden',
      }}
    >
      {/* Main row */}
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '40px 2fr 1fr 120px',
          alignItems: 'center',
          padding: theme.spacing.md,
          gap: theme.spacing.md,
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
          },
        }}
        onClick={onToggleExpand}
      >
        <div css={{ display: 'flex', alignItems: 'center' }}>
          {isExpanded ? (
            <ChevronDownIcon css={{ fontSize: 20, color: theme.colors.textSecondary }} />
          ) : (
            <ChevronRightIcon css={{ fontSize: 20, color: theme.colors.textSecondary }} />
          )}
        </div>

        <div css={{ overflow: 'hidden' }}>
          <Typography.Text css={{ fontWeight: 600, fontSize: theme.typography.fontSizeMd }}>
            {secret.secret_name}
          </Typography.Text>
          <Typography.Text
            css={{
              display: 'block',
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              marginTop: theme.spacing.xs,
            }}
          >
            {secret.masked_value}
          </Typography.Text>
        </div>

        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage defaultMessage="Usage:" description="Secret management drawer > usage label" />
          </Typography.Text>
          <div
            css={{
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: hasZeroUsage
                ? theme.colors.backgroundValidationDanger
                : theme.colors.backgroundSecondary,
              border: `1px solid ${hasZeroUsage ? theme.colors.borderValidationDanger : theme.colors.border}`,
            }}
          >
            <Typography.Text
              css={{
                color: hasZeroUsage ? theme.colors.textValidationDanger : theme.colors.textPrimary,
                fontWeight: 600,
                fontSize: theme.typography.fontSizeMd,
              }}
            >
              {isExpanded && isLoadingBindings ? '...' : usageCount}
            </Typography.Text>
          </div>
        </div>

        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.secrets.management_drawer.update_button"
            size="small"
            icon={<PencilIcon />}
            onClick={(e) => {
              e.stopPropagation();
              onUpdate();
            }}
            aria-label={intl.formatMessage({
              defaultMessage: 'Update secret',
              description: 'Secret management drawer > update button aria label',
            })}
          >
            <FormattedMessage defaultMessage="Update" description="Secret management drawer > update button" />
          </Button>
          <Button
            componentId="mlflow.secrets.management_drawer.delete_button"
            size="small"
            icon={<TrashIcon />}
            danger={!hasZeroUsage}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            aria-label={intl.formatMessage({
              defaultMessage: 'Delete secret',
              description: 'Secret management drawer > delete button aria label',
            })}
          >
            <FormattedMessage defaultMessage="Delete" description="Secret management drawer > delete button" />
          </Button>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div
          css={{
            padding: theme.spacing.lg,
            paddingTop: 0,
            backgroundColor: theme.colors.backgroundSecondary,
            borderTop: `1px solid ${theme.colors.border}`,
          }}
        >
          {isLoadingBindings ? (
            <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
              <Spinner />
            </div>
          ) : bindings.length === 0 ? (
            <div
              css={{
                padding: theme.spacing.lg,
                textAlign: 'center',
                backgroundColor: theme.colors.backgroundPrimary,
                borderRadius: theme.borders.borderRadiusMd,
                border: `1px dashed ${theme.colors.border}`,
              }}
            >
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="This secret is not currently being used by any resources or routes."
                  description="Secret management drawer > no bindings message"
                />
              </Typography.Text>
            </div>
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
              {/* Routes section */}
              {routeNames.length > 0 && (
                <div>
                  <Typography.Title level={5} css={{ marginTop: theme.spacing.md, marginBottom: theme.spacing.sm }}>
                    <FormattedMessage
                      defaultMessage="Routes ({count})"
                      description="Secret management drawer > routes section title"
                      values={{ count: routeNames.length }}
                    />
                  </Typography.Title>
                  <div
                    css={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                      gap: theme.spacing.sm,
                    }}
                  >
                    {routeNames.map((routeName, index) => (
                      <div
                        key={index}
                        css={{
                          padding: theme.spacing.sm,
                          backgroundColor: theme.colors.backgroundPrimary,
                          borderRadius: theme.borders.borderRadiusMd,
                          border: `1px solid ${theme.colors.border}`,
                        }}
                      >
                        <Typography.Text css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}>
                          {routeName}
                        </Typography.Text>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Resource bindings section */}
              <div>
                <Typography.Title level={5} css={{ marginTop: theme.spacing.md, marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Resource Bindings ({count})"
                    description="Secret management drawer > bindings section title"
                    values={{ count: bindings.length }}
                  />
                </Typography.Title>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                  {bindings.map((binding) => (
                    <div
                      key={binding.binding_id}
                      css={{
                        padding: theme.spacing.md,
                        backgroundColor: theme.colors.backgroundPrimary,
                        borderRadius: theme.borders.borderRadiusMd,
                        border: `1px solid ${theme.colors.border}`,
                        display: 'grid',
                        gridTemplateColumns: '150px 1fr 200px',
                        gap: theme.spacing.md,
                        alignItems: 'center',
                      }}
                    >
                      <div>
                        <Typography.Text
                          size="sm"
                          color="secondary"
                          css={{ display: 'block', marginBottom: theme.spacing.xs }}
                        >
                          <FormattedMessage
                            defaultMessage="Resource Type"
                            description="Secret management drawer > binding resource type label"
                          />
                        </Typography.Text>
                        <Typography.Text css={{ fontWeight: 500 }}>{formatResourceType(binding)}</Typography.Text>
                      </div>
                      <div css={{ overflow: 'hidden' }}>
                        <Typography.Text
                          size="sm"
                          color="secondary"
                          css={{ display: 'block', marginBottom: theme.spacing.xs }}
                        >
                          <FormattedMessage
                            defaultMessage="Resource"
                            description="Secret management drawer > binding resource label"
                          />
                        </Typography.Text>
                        <Typography.Text
                          css={{
                            fontFamily: 'monospace',
                            fontSize: theme.typography.fontSizeSm,
                            wordBreak: 'break-all',
                          }}
                        >
                          {getResourceDisplay(binding)}
                        </Typography.Text>
                      </div>
                      <div>
                        <Typography.Text
                          size="sm"
                          color="secondary"
                          css={{ display: 'block', marginBottom: theme.spacing.xs }}
                        >
                          <FormattedMessage
                            defaultMessage="Environment Variable"
                            description="Secret management drawer > binding field label"
                          />
                        </Typography.Text>
                        <Typography.Text
                          css={{
                            fontFamily: 'monospace',
                            fontSize: theme.typography.fontSizeSm,
                            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                            backgroundColor: theme.colors.backgroundSecondary,
                            borderRadius: theme.borders.borderRadiusSm,
                            display: 'inline-block',
                          }}
                        >
                          {binding.field_name}
                        </Typography.Text>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
