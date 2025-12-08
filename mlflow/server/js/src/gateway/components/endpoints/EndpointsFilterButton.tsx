import { ProviderFilterButton, type ProviderFilter } from '../shared/ProviderFilterButton';

// Re-export the filter type with the original name for backwards compatibility
export type EndpointsFilter = ProviderFilter;

interface EndpointsFilterButtonProps {
  availableProviders: string[];
  filter: EndpointsFilter;
  onFilterChange: (filter: EndpointsFilter) => void;
}

/**
 * Filter button for the endpoints list.
 * Thin wrapper around the shared ProviderFilterButton.
 */
export const EndpointsFilterButton = ({ availableProviders, filter, onFilterChange }: EndpointsFilterButtonProps) => {
  return (
    <ProviderFilterButton
      availableProviders={availableProviders}
      filter={filter}
      onFilterChange={onFilterChange}
      componentIdPrefix="mlflow.gateway.endpoints-list"
    />
  );
};
