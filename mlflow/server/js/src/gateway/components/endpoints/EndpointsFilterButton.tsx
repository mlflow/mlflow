import { useState, useMemo } from 'react';
import {
  Button,
  ChevronDownIcon,
  FilterIcon,
  Popover,
  Checkbox,
  XCircleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { formatProviderName } from '../../utils/providerUtils';

export interface EndpointsFilter {
  providers: string[];
}

interface EndpointsFilterButtonProps {
  availableProviders: string[];
  filter: EndpointsFilter;
  onFilterChange: (filter: EndpointsFilter) => void;
}

export const EndpointsFilterButton = ({ availableProviders, filter, onFilterChange }: EndpointsFilterButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isOpen, setIsOpen] = useState(false);

  const hasActiveFilters = filter.providers.length > 0;
  const filterCount = filter.providers.length;

  const sortedProviders = useMemo(() => {
    return [...availableProviders].sort((a, b) => formatProviderName(a).localeCompare(formatProviderName(b)));
  }, [availableProviders]);

  const handleProviderToggle = (provider: string) => {
    const newProviders = filter.providers.includes(provider)
      ? filter.providers.filter((p) => p !== provider)
      : [...filter.providers, provider];
    onFilterChange({ ...filter, providers: newProviders });
  };

  const handleClearFilters = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFilterChange({ providers: [] });
  };

  return (
    <Popover.Root componentId="mlflow.gateway.endpoints-list.filter-popover" open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <Button
          componentId="mlflow.gateway.endpoints-list.filter-button"
          endIcon={<ChevronDownIcon />}
          css={{
            border: hasActiveFilters ? `1px solid ${theme.colors.actionDefaultBorderFocus} !important` : '',
            backgroundColor: hasActiveFilters ? `${theme.colors.actionDefaultBackgroundHover} !important` : '',
          }}
        >
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <FilterIcon />
            {intl.formatMessage(
              {
                defaultMessage: 'Filter{count}',
                description: 'Filter button label with count',
              },
              { count: hasActiveFilters ? ` (${filterCount})` : '' },
            )}
            {hasActiveFilters && (
              <XCircleFillIcon
                css={{
                  fontSize: 12,
                  cursor: 'pointer',
                  color: theme.colors.grey400,
                  '&:hover': { color: theme.colors.grey600 },
                }}
                onClick={handleClearFilters}
              />
            )}
          </div>
        </Button>
      </Popover.Trigger>
      <Popover.Content align="start" css={{ padding: theme.spacing.md, minWidth: 200 }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <div css={{ fontWeight: theme.typography.typographyBoldFontWeight, marginBottom: theme.spacing.xs }}>
            <FormattedMessage defaultMessage="Provider" description="Filter section label for provider" />
          </div>
          {sortedProviders.length === 0 ? (
            <div css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage defaultMessage="No providers available" description="Empty state for provider filter" />
            </div>
          ) : (
            sortedProviders.map((provider) => (
              <Checkbox
                key={provider}
                componentId={`mlflow.gateway.endpoints-list.filter.provider.${provider}`}
                isChecked={filter.providers.includes(provider)}
                onChange={() => handleProviderToggle(provider)}
              >
                {formatProviderName(provider)}
              </Checkbox>
            ))
          )}
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
