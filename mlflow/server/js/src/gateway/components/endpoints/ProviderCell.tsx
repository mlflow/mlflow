import { useState } from 'react';
import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { formatProviderName } from '../../utils/providerUtils';
import type { Endpoint } from '../../types';

interface ProviderCellProps {
  modelMappings: Endpoint['model_mappings'];
}

export const ProviderCell = ({ modelMappings }: ProviderCellProps) => {
  const { theme } = useDesignSystemTheme();
  const [isExpanded, setIsExpanded] = useState(false);

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

  const displayedProviders = isExpanded ? uniqueProviders : [primaryProvider];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs / 2 }}>
      {displayedProviders.map((provider) => (
        <Tag key={provider} componentId="mlflow.gateway.endpoints-list.provider-tag">
          {formatProviderName(provider)}
        </Tag>
      ))}
      {additionalCount > 0 && (
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          css={{
            background: 'none',
            border: 'none',
            padding: 0,
            margin: 0,
            textAlign: 'left',
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
            cursor: 'pointer',
            '&:hover': { textDecoration: 'underline' },
          }}
        >
          {isExpanded ? 'Show less' : `+${additionalCount} more`}
        </button>
      )}
    </div>
  );
};
