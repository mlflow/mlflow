import {
  Button,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowAction,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useListBindings } from '../hooks/useListBindings';
import { useCallback } from 'react';
import type { SecretBinding } from '../types';

export interface SecretBindingsListProps {
  secretId: string;
  variant?: 'default' | 'warning';
  isSharedSecret?: boolean;
  onUnbind?: (binding: SecretBinding) => void;
}

export const SecretBindingsList = ({
  secretId,
  variant = 'default',
  isSharedSecret = false,
  onUnbind
}: SecretBindingsListProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { bindings, isLoading, error } = useListBindings({ secretId });

  const formatResourceType = (resourceType: string) => {
    // Convert SCORER_JOB to "Scorer Job", GLOBAL to "Global", etc.
    return resourceType
      .split('_')
      .map(word => word.charAt(0) + word.slice(1).toLowerCase())
      .join(' ');
  };

  const handleUnbind = useCallback((binding: SecretBinding) => {
    onUnbind?.(binding);
  }, [onUnbind]);

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
      <Table scrollable>
        <TableRow isHeader>
          <TableHeader componentId="mlflow.secrets.bindings_list.resource_type">
            <FormattedMessage
              defaultMessage="Resource Type"
              description="Secret bindings list > resource type column header"
            />
          </TableHeader>
          <TableHeader componentId="mlflow.secrets.bindings_list.resource">
            <FormattedMessage
              defaultMessage="Resource"
              description="Secret bindings list > resource column header"
            />
          </TableHeader>
          <TableHeader componentId="mlflow.secrets.bindings_list.field_name">
            <FormattedMessage
              defaultMessage="Environment Variable"
              description="Secret bindings list > field name column header"
            />
          </TableHeader>
          {isSharedSecret && <TableHeader componentId="mlflow.secrets.bindings_list.actions" />}
        </TableRow>
        {bindings.map((binding) => (
          <TableRow key={binding.binding_id}>
            <TableCell>
              <Typography.Text>{formatResourceType(binding.resource_type)}</Typography.Text>
            </TableCell>
            <TableCell>
              <Tooltip
                componentId={`mlflow.secrets.bindings_list.resource_tooltip.${binding.binding_id}`}
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
            </TableCell>
            <TableCell>
              <Typography.Text
                css={{
                  fontFamily: 'monospace',
                  fontSize: theme.typography.fontSizeSm,
                }}
              >
                {binding.field_name}
              </Typography.Text>
            </TableCell>
            {isSharedSecret && (
              <TableRowAction>
                {canUnbind(binding) && (
                  <Button
                    componentId="mlflow.secrets.bindings_list.unbind_button"
                    size="small"
                    icon={<TrashIcon />}
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Unbind',
                      description: 'Secret bindings list > unbind action label',
                    })}
                    onClick={() => handleUnbind(binding)}
                    css={{ padding: '4px' }}
                  />
                )}
              </TableRowAction>
            )}
          </TableRow>
        ))}
      </Table>
    </div>
  );
};
