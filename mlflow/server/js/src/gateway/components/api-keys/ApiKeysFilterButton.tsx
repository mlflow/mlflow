import { ProviderFilterButton, type ProviderFilter } from '../shared/ProviderFilterButton';

// Re-export the filter type with the original name for backwards compatibility
export type ApiKeysFilter = ProviderFilter;

interface ApiKeysFilterButtonProps {
  availableProviders: string[];
  filter: ApiKeysFilter;
  onFilterChange: (filter: ApiKeysFilter) => void;
}

/**
 * Filter button for the API keys list.
 * Thin wrapper around the shared ProviderFilterButton.
 */
export const ApiKeysFilterButton = ({ availableProviders, filter, onFilterChange }: ApiKeysFilterButtonProps) => {
  return (
    <ProviderFilterButton
      availableProviders={availableProviders}
      filter={filter}
      onFilterChange={onFilterChange}
      componentIdPrefix="mlflow.gateway.api-keys-list"
    />
  );
};
