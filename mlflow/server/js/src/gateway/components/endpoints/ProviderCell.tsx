import { Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { formatProviderName } from '../../utils/providerUtils';
import type { Endpoint } from '../../types';

interface ProviderCellProps {
  modelMappings: Endpoint['model_mappings'];
}

export const ProviderCell = ({ modelMappings }: ProviderCellProps) => {
  const { theme } = useDesignSystemTheme();

  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  // Get unique providers from all model mappings
  const allProviders = modelMappings
    .map((m) => m.model_definition?.provider)
    .filter((provider): provider is string => Boolean(provider));

  const uniqueProviders = [...new Set(allProviders)];

  if (uniqueProviders.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryProvider = uniqueProviders[0];
  const additionalProviders = uniqueProviders.slice(1);
  const additionalCount = additionalProviders.length;

  const tooltipContent =
    additionalCount > 0 ? additionalProviders.map((p) => formatProviderName(p)).join(', ') : undefined;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs / 2 }}>
      <Tag componentId="mlflow.gateway.endpoints-list.provider-tag">{formatProviderName(primaryProvider)}</Tag>
      {additionalCount > 0 && (
        <Tooltip componentId="mlflow.gateway.endpoints-list.providers-more-tooltip" content={tooltipContent}>
          <button
            type="button"
            css={{
              background: 'none',
              border: 'none',
              padding: 0,
              margin: 0,
              textAlign: 'left',
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              cursor: 'default',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            +{additionalCount} more
          </button>
        </Tooltip>
      )}
    </div>
  );
};
