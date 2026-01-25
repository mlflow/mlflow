import { useMemo, useState, useCallback } from 'react';
import { SimpleSelect, SimpleSelectOption, Typography, useDesignSystemTheme, Button, CloseIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { createTraceTagFilter, GatewayTraceTagKey } from '../../shared/web-shared/model-trace-explorer';

export interface GatewayUsageFiltersProps {
  /** Available providers to filter by */
  providers: string[];
  /** Available models to filter by */
  models: string[];
  /** Callback when filters change */
  onFiltersChange: (filters: string[]) => void;
  /** Whether the filter controls are disabled */
  disabled?: boolean;
}

/**
 * Component for filtering gateway usage charts by provider and model.
 * Provides dropdowns for selecting provider and model filters.
 */
export const GatewayUsageFilters = ({
  providers,
  models,
  onFiltersChange,
  disabled = false,
}: GatewayUsageFiltersProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  // Build filter expressions from selected values
  const updateFilters = useCallback(
    (provider: string | null, model: string | null) => {
      const filters: string[] = [];
      if (provider) {
        filters.push(createTraceTagFilter(GatewayTraceTagKey.PROVIDER, provider));
      }
      if (model) {
        filters.push(createTraceTagFilter(GatewayTraceTagKey.MODEL, model));
      }
      onFiltersChange(filters);
    },
    [onFiltersChange],
  );

  const handleProviderChange = useCallback(
    (value: string | null) => {
      setSelectedProvider(value);
      updateFilters(value, selectedModel);
    },
    [selectedModel, updateFilters],
  );

  const handleModelChange = useCallback(
    (value: string | null) => {
      setSelectedModel(value);
      updateFilters(selectedProvider, value);
    },
    [selectedProvider, updateFilters],
  );

  const clearAllFilters = useCallback(() => {
    setSelectedProvider(null);
    setSelectedModel(null);
    onFiltersChange([]);
  }, [onFiltersChange]);

  const hasActiveFilters = selectedProvider !== null || selectedModel !== null;

  // Sort providers and models alphabetically
  const sortedProviders = useMemo(() => [...providers].sort(), [providers]);
  const sortedModels = useMemo(() => [...models].sort(), [models]);

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      {/* Provider filter */}
      {sortedProviders.length > 0 && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Provider:" description="Provider filter label" />
          </Typography.Text>
          <SimpleSelect
            id="gateway-usage-provider-filter"
            componentId="mlflow.gateway.usage.provider-filter"
            value={selectedProvider ?? ''}
            onChange={({ target }) => handleProviderChange(target.value || null)}
            css={{ minWidth: 150 }}
            disabled={disabled}
          >
            <SimpleSelectOption value="">
              <FormattedMessage defaultMessage="All providers" description="All providers option" />
            </SimpleSelectOption>
            {sortedProviders.map((provider) => (
              <SimpleSelectOption key={provider} value={provider}>
                {provider}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        </div>
      )}

      {/* Model filter */}
      {sortedModels.length > 0 && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Model:" description="Model filter label" />
          </Typography.Text>
          <SimpleSelect
            id="gateway-usage-model-filter"
            componentId="mlflow.gateway.usage.model-filter"
            value={selectedModel ?? ''}
            onChange={({ target }) => handleModelChange(target.value || null)}
            css={{ minWidth: 200 }}
            disabled={disabled}
          >
            <SimpleSelectOption value="">
              <FormattedMessage defaultMessage="All models" description="All models option" />
            </SimpleSelectOption>
            {sortedModels.map((model) => (
              <SimpleSelectOption key={model} value={model}>
                {model}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        </div>
      )}

      {/* Clear filters button */}
      {hasActiveFilters && (
        <Button
          componentId="mlflow.gateway.usage.clear-filters"
          size="small"
          type="tertiary"
          onClick={clearAllFilters}
          disabled={disabled}
          icon={<CloseIcon />}
        >
          <FormattedMessage defaultMessage="Clear filters" description="Clear filters button" />
        </Button>
      )}
    </div>
  );
};
