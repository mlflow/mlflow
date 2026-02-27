import { ProviderFilterButton, type ProviderFilter } from '../shared/ProviderFilterButton';

export type ApiKeysFilter = ProviderFilter;

interface ApiKeysFilterButtonProps {
  availableProviders: string[];
  filter: ApiKeysFilter;
  onFilterChange: (filter: ApiKeysFilter) => void;
}

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
