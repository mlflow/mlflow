import {
  Button,
  Empty,
  Spinner,
  TrashIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCallback } from 'react';
import type { SecretBinding } from '../types';
import { useListBindings } from '../hooks/useListBindings';

export interface BindingsTableProps {
  secretId: string;
  variant?: 'default' | 'warning' | 'compact';
  isSharedSecret?: boolean;
  onUnbind?: (binding: SecretBinding) => void;
  maxHeight?: number;
}

const formatResourceType = (resourceType: string) => {
  // Convert SCORER_JOB to "Scorer Job", GLOBAL to "Global", etc.
  return resourceType
    .split('_')
    .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
    .join(' ');
};

/**
 * A reusable component for displaying secret bindings in a clean, scrollable table.
 * Handles large numbers of bindings efficiently with proper alignment and styling.
 */
export const BindingsTable = ({
  secretId,
  variant = 'default',
  isSharedSecret = false,
  onUnbind,
  maxHeight = 400,
}: BindingsTableProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { bindings, isLoading, error } = useListBindings({ secretId });

  const handleUnbind = useCallback(
    (binding: SecretBinding) => {
      onUnbind?.(binding);
    },
    [onUnbind],
  );

  // Only show unbind for shared/global secrets bound to actual resources (not GLOBAL type)
  const canUnbind = (binding: SecretBinding) => {
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
          description="Bindings table > error loading bindings"
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
            description="Bindings table > no bindings"
          />
        }
      />
    );
  }

  const isCompact = variant === 'compact';
  const showActions = isSharedSecret && bindings.some(canUnbind);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {variant === 'warning' && (
        <Typography.Text css={{ color: theme.colors.textValidationWarning }}>
          <FormattedMessage
            defaultMessage="The following resources are currently using this secret:"
            description="Bindings table > warning message"
          />
        </Typography.Text>
      )}

      <div
        css={{
          maxHeight: `${maxHeight}px`,
          overflow: 'auto',
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
        }}
      >
        <table
          css={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: isCompact ? theme.typography.fontSizeSm : theme.typography.fontSizeBase,
          }}
        >
          <thead
            css={{
              position: 'sticky',
              top: 0,
              backgroundColor: theme.colors.backgroundPrimary,
              borderBottom: `1px solid ${theme.colors.border}`,
              zIndex: 1,
            }}
          >
            <tr>
              <th
                css={{
                  textAlign: 'left',
                  padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                  fontWeight: theme.typography.typographyBoldFontWeight,
                  color: theme.colors.textSecondary,
                }}
              >
                <FormattedMessage
                  defaultMessage="Resource Type"
                  description="Bindings table > resource type column header"
                />
              </th>
              <th
                css={{
                  textAlign: 'left',
                  padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                  fontWeight: theme.typography.typographyBoldFontWeight,
                  color: theme.colors.textSecondary,
                }}
              >
                <FormattedMessage defaultMessage="Resource" description="Bindings table > resource column header" />
              </th>
              <th
                css={{
                  textAlign: 'left',
                  padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                  fontWeight: theme.typography.typographyBoldFontWeight,
                  color: theme.colors.textSecondary,
                }}
              >
                <FormattedMessage
                  defaultMessage="Environment Variable"
                  description="Bindings table > field name column header"
                />
              </th>
              {showActions && (
                <th
                  css={{
                    textAlign: 'right',
                    padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                    color: theme.colors.textSecondary,
                    width: '80px',
                  }}
                >
                  <FormattedMessage defaultMessage="Actions" description="Bindings table > actions column header" />
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {bindings.map((binding) => (
              <tr
                key={binding.binding_id}
                css={{
                  borderBottom: `1px solid ${theme.colors.border}`,
                  '&:last-child': {
                    borderBottom: 'none',
                  },
                  '&:hover': {
                    backgroundColor: theme.colors.backgroundSecondary,
                  },
                  // Hide the unbind button by default, show on row hover
                  '&:hover .unbind-button': {
                    opacity: 1,
                  },
                }}
              >
                <td
                  css={{
                    textAlign: 'left',
                    padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                    verticalAlign: 'middle',
                  }}
                >
                  {formatResourceType(binding.resource_type)}
                </td>
                <td
                  css={{
                    textAlign: 'left',
                    padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                    verticalAlign: 'middle',
                    maxWidth: '300px',
                  }}
                >
                  <Tooltip
                    componentId={`mlflow.bindings_table.resource_tooltip.${binding.binding_id}`}
                    content={binding.resource_id}
                  >
                    <Typography.Text
                      ellipsis
                      css={{
                        fontFamily: 'monospace',
                        fontSize: theme.typography.fontSizeSm,
                      }}
                    >
                      {binding.resource_id}
                    </Typography.Text>
                  </Tooltip>
                </td>
                <td
                  css={{
                    textAlign: 'left',
                    padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                    verticalAlign: 'middle',
                  }}
                >
                  <Typography.Text
                    css={{
                      fontFamily: 'monospace',
                      fontSize: theme.typography.fontSizeSm,
                    }}
                  >
                    {binding.field_name}
                  </Typography.Text>
                </td>
                {showActions && (
                  <td
                    css={{
                      textAlign: 'right',
                      padding: isCompact ? theme.spacing.sm : theme.spacing.md,
                      verticalAlign: 'middle',
                    }}
                  >
                    {canUnbind(binding) && (
                      <Button
                        componentId="mlflow.bindings_table.unbind_button"
                        className="unbind-button"
                        size="small"
                        icon={<TrashIcon />}
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Unbind',
                          description: 'Bindings table > unbind action label',
                        })}
                        onClick={() => handleUnbind(binding)}
                        css={{
                          opacity: 0,
                          transition: 'opacity 0.2s ease-in-out',
                        }}
                      />
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
