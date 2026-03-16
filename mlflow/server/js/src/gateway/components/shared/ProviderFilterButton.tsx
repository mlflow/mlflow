import { useState, useMemo, useCallback } from 'react';
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

export interface ProviderFilter {
  providers: string[];
}

interface ProviderFilterButtonProps {
  availableProviders: string[];
  filter: ProviderFilter;
  onFilterChange: (filter: ProviderFilter) => void;
  componentIdPrefix: string;
}

export const ProviderFilterButton = ({
  availableProviders,
  filter,
  onFilterChange,
  componentIdPrefix,
}: ProviderFilterButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isOpen, setIsOpen] = useState(false);

  const hasActiveFilters = filter.providers.length > 0;
  const filterCount = filter.providers.length;

  const sortedProviders = useMemo(() => {
    return [...availableProviders].sort((a, b) => formatProviderName(a).localeCompare(formatProviderName(b)));
  }, [availableProviders]);

  const handleProviderToggle = useCallback(
    (provider: string) => {
      const newProviders = filter.providers.includes(provider)
        ? filter.providers.filter((p) => p !== provider)
        : [...filter.providers, provider];
      onFilterChange({ ...filter, providers: newProviders });
    },
    [filter, onFilterChange],
  );

  const handleClearFilters = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onFilterChange({ providers: [] });
    },
    [onFilterChange],
  );

  return (
    <Popover.Root componentId={`${componentIdPrefix}.filter-popover`} open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <Button
          componentId={`${componentIdPrefix}.filter-button`}
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
                defaultMessage: 'Provider{count}',
                description: 'Provider filter button label with count',
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
                componentId={`${componentIdPrefix}.filter.provider.${provider}`}
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
