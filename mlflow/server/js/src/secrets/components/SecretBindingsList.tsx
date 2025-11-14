import {
  Button,
  Empty,
  Spinner,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useListBindings } from '../hooks/useListBindings';
import { useCallback } from 'react';

export interface SecretBindingsListProps {
  secretId: string;
  variant?: 'default' | 'warning';
  isSharedSecret?: boolean;
  onUnbind?: (bindingId: string) => void;
}

export const SecretBindingsList = ({
  secretId,
  variant = 'default',
  isSharedSecret = false,
  onUnbind,
}: SecretBindingsListProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { bindings, isLoading, error } = useListBindings({ secretId });

  const formatResourceType = (binding: any) => {
    // If this is a route binding, show "Route" regardless of the resource_type field
    if (binding.route_id) {
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
    if (binding.route_id) {
      return binding.route_name || binding.route_id;
    }
    return binding.resource_id;
  };

  const handleUnbind = useCallback(
    (bindingId: string) => {
      onUnbind?.(bindingId);
    },
    [onUnbind],
  );

  // Only show unbind for shared/global secrets bound to actual resources (not GLOBAL type)
  const canUnbind = (binding: any) => {
    return isSharedSecret && binding.resource_type !== 'GLOBAL';
  };

  if (isLoading) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.md }}>
        <Spinner />
      </div>
    );
  }

  if (error) {
    return (
      <Typography.Text color="error">
        <FormattedMessage
          defaultMessage="Failed to load bindings"
          description="Secret bindings list > error loading bindings"
        />
      </Typography.Text>
    );
  }

  if (bindings.length === 0) {
    return (
      <Empty
        description={
          <FormattedMessage
            defaultMessage="No bindings found for this secret"
            description="Secret bindings list > no bindings"
          />
        }
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {variant === 'warning' && (
        <Typography.Text css={{ color: theme.colors.textValidationWarning }}>
          <FormattedMessage
            defaultMessage="The following resources are currently using this secret:"
            description="Secret bindings list > warning message"
          />
        </Typography.Text>
      )}
      <div
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: isSharedSecret ? '150px 1fr 200px 80px' : '150px 1fr 200px',
            backgroundColor: theme.colors.backgroundSecondary,
            borderBottom: `1px solid ${theme.colors.border}`,
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            fontWeight: theme.typography.typographyBoldFontWeight,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
          }}
        >
          <div>
            <FormattedMessage
              defaultMessage="Resource Type"
              description="Secret bindings list > resource type column header"
            />
          </div>
          <div>
            <FormattedMessage defaultMessage="Resource" description="Secret bindings list > resource column header" />
          </div>
          <div>
            <FormattedMessage
              defaultMessage="Environment Variable"
              description="Secret bindings list > field name column header"
            />
          </div>
          {isSharedSecret && <div />}
        </div>
        {/* Rows */}
        {bindings.map((binding, index) => (
          <div
            key={binding.binding_id}
            css={{
              display: 'grid',
              gridTemplateColumns: isSharedSecret ? '150px 1fr 200px 80px' : '150px 1fr 200px',
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              borderBottom: index < bindings.length - 1 ? `1px solid ${theme.colors.border}` : 'none',
              '&:hover': {
                backgroundColor: theme.colors.backgroundSecondary,
              },
              alignItems: 'center',
            }}
          >
            <div>
              <Typography.Text>{formatResourceType(binding)}</Typography.Text>
            </div>
            <div css={{ overflow: 'hidden' }}>
              <Tooltip
                componentId={`mlflow.secrets.bindings_list.resource_tooltip.${binding.binding_id}`}
                content={getResourceDisplay(binding)}
              >
                <Typography.Text
                  ellipsis
                  css={{
                    fontFamily: 'monospace',
                    fontSize: theme.typography.fontSizeSm,
                  }}
                >
                  {getResourceDisplay(binding)}
                </Typography.Text>
              </Tooltip>
            </div>
            <div>
              <Typography.Text
                css={{
                  fontFamily: 'monospace',
                  fontSize: theme.typography.fontSizeSm,
                }}
              >
                {binding.field_name}
              </Typography.Text>
            </div>
            {isSharedSecret && (
              <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
                {canUnbind(binding) && (
                  <Button
                    componentId="mlflow.secrets.bindings_list.unbind_button"
                    size="small"
                    icon={<TrashIcon />}
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Unbind',
                      description: 'Secret bindings list > unbind action label',
                    })}
                    onClick={() => handleUnbind(binding.binding_id)}
                    css={{ padding: '4px' }}
                  />
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
