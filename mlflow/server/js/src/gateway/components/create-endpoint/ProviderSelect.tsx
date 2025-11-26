import {
  TypeaheadComboboxRoot,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxSectionHeader,
  useComboboxState,
  useDesignSystemTheme,
  FormUI,
  Spinner,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useProvidersQuery } from '../../hooks/useProvidersQuery';
import { groupProviders, formatProviderName } from '../../utils/providerUtils';
import { useMemo, useState, useCallback } from 'react';

interface ProviderSelectProps {
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  /** Component ID prefix for telemetry (default: 'mlflow.gateway.provider-select') */
  componentIdPrefix?: string;
}

interface ProviderItem {
  provider: string;
  displayName: string;
  group: 'common' | 'other';
}

export const ProviderSelect = ({
  value,
  onChange,
  disabled,
  error,
  componentIdPrefix = 'mlflow.gateway.provider-select',
}: ProviderSelectProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { data: providers, isLoading } = useProvidersQuery();

  // Create provider items with display names and group info
  const allProviderItems = useMemo((): ProviderItem[] => {
    if (!providers) return [];

    const { common, other } = groupProviders(providers);

    const commonItems: ProviderItem[] = common.map((p) => ({
      provider: p,
      displayName: formatProviderName(p),
      group: 'common' as const,
    }));

    const otherItems: ProviderItem[] = other.map((p) => ({
      provider: p,
      displayName: formatProviderName(p),
      group: 'other' as const,
    }));

    return [...commonItems, ...otherItems];
  }, [providers]);

  // State for filtered items (null included for combobox state compatibility)
  const [filteredItems, setFilteredItems] = useState<(ProviderItem | null)[]>(allProviderItems);

  // Update filtered items when allProviderItems changes
  useMemo(() => {
    setFilteredItems(allProviderItems);
  }, [allProviderItems]);

  // Find the currently selected item
  const selectedItem = useMemo(() => {
    return allProviderItems.find((item) => item.provider === value) ?? null;
  }, [allProviderItems, value]);

  // Matcher function for filtering
  const matcher = useCallback((item: ProviderItem | null, query: string): boolean => {
    if (!item) return false;
    const lowerQuery = query.toLowerCase();
    return item.displayName.toLowerCase().includes(lowerQuery) || item.provider.toLowerCase().includes(lowerQuery);
  }, []);

  // Handle selection change
  const handleChange = useCallback(
    (item: ProviderItem | null) => {
      if (item) {
        onChange(item.provider);
      }
    },
    [onChange],
  );

  const comboboxState = useComboboxState<ProviderItem | null>({
    componentId: componentIdPrefix,
    allItems: allProviderItems,
    items: filteredItems,
    setItems: setFilteredItems,
    multiSelect: false,
    itemToString: (item) => item?.displayName ?? '',
    matcher,
    formValue: selectedItem,
    formOnChange: handleChange,
    initialInputValue: selectedItem?.displayName ?? '',
  });

  // Group filtered items for rendering (filter out nulls)
  const groupedItems = useMemo(() => {
    const nonNullItems = filteredItems.filter((item): item is ProviderItem => item !== null);
    const common = nonNullItems.filter((item) => item.group === 'common');
    const other = nonNullItems.filter((item) => item.group === 'other');
    return { common, other };
  }, [filteredItems]);

  if (isLoading) {
    return (
      <div>
        <FormUI.Label htmlFor={componentIdPrefix}>
          <FormattedMessage defaultMessage="Provider" description="Label for provider select field" />
        </FormUI.Label>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginTop: theme.spacing.xs }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading providers..." description="Loading message for providers" />
        </div>
      </div>
    );
  }

  // Build the menu items with section headers
  const menuItems: JSX.Element[] = [];
  let itemIndex = 0;

  if (groupedItems.common.length > 0) {
    menuItems.push(
      <TypeaheadComboboxSectionHeader key="common-header">
        {intl.formatMessage({ defaultMessage: 'Common Providers', description: 'Section header for common providers' })}
      </TypeaheadComboboxSectionHeader>,
    );

    groupedItems.common.forEach((item) => {
      menuItems.push(
        <TypeaheadComboboxMenuItem
          key={item.provider}
          item={item}
          index={itemIndex}
          comboboxState={comboboxState}
          data-testid={`${componentIdPrefix}.option.${item.provider}`}
        >
          {item.displayName}
        </TypeaheadComboboxMenuItem>,
      );
      itemIndex++;
    });
  }

  if (groupedItems.other.length > 0) {
    menuItems.push(
      <TypeaheadComboboxSectionHeader key="other-header">
        {intl.formatMessage({ defaultMessage: 'Other Providers', description: 'Section header for other providers' })}
      </TypeaheadComboboxSectionHeader>,
    );

    groupedItems.other.forEach((item) => {
      menuItems.push(
        <TypeaheadComboboxMenuItem
          key={item.provider}
          item={item}
          index={itemIndex}
          comboboxState={comboboxState}
          data-testid={`${componentIdPrefix}.option.${item.provider}`}
        >
          {item.displayName}
        </TypeaheadComboboxMenuItem>,
      );
      itemIndex++;
    });
  }

  return (
    <div css={{ minWidth: 300 }}>
      <FormUI.Label htmlFor={componentIdPrefix}>
        <FormattedMessage defaultMessage="Provider" description="Label for provider select field" />
      </FormUI.Label>
      <TypeaheadComboboxRoot id={componentIdPrefix} comboboxState={comboboxState}>
        <TypeaheadComboboxInput
          placeholder={intl.formatMessage({
            defaultMessage: 'Search for a provider...',
            description: 'Placeholder for provider search input',
          })}
          comboboxState={comboboxState}
          formOnChange={handleChange}
          validationState={error ? 'error' : undefined}
          disabled={disabled}
          allowClear
          showComboboxToggleButton
        />
        <TypeaheadComboboxMenu comboboxState={comboboxState} matchTriggerWidth minWidth={300}>
          {menuItems}
        </TypeaheadComboboxMenu>
      </TypeaheadComboboxRoot>
      {error && <FormUI.Message type="error" message={error} />}
    </div>
  );
};
