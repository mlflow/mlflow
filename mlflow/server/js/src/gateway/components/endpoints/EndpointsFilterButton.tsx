import { ProviderFilterButton, type ProviderFilter } from '../shared/ProviderFilterButton';

export type EndpointsFilter = ProviderFilter;

interface EndpointsFilterButtonProps {
  availableProviders: string[];
  filter: EndpointsFilter;
  onFilterChange: (filter: EndpointsFilter) => void;
}

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
