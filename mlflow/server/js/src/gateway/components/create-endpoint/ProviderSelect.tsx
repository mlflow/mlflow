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
import { useMemo, useCallback } from 'react';

interface ProviderSelectProps {
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
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

  // Find the currently selected item
  const selectedItem = useMemo(() => {
    return allProviderItems.find((item) => item.provider === value) ?? null;
  }, [allProviderItems, value]);

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
    items: allProviderItems,
    setItems: () => {},
    multiSelect: false,
    itemToString: (item) => item?.displayName ?? '',
    matcher: (item, query) => {
      if (!item) return false;
      const lowerQuery = query.toLowerCase();
      return item.displayName.toLowerCase().includes(lowerQuery) || item.provider.toLowerCase().includes(lowerQuery);
    },
    formValue: selectedItem,
    formOnChange: handleChange,
    initialInputValue: selectedItem?.displayName ?? '',
  });

  // Group items for rendering with section headers
  const groupedItems = useMemo(() => {
    const common = allProviderItems.filter((item) => item.group === 'common');
    const other = allProviderItems.filter((item) => item.group === 'other');
    return { common, other };
  }, [allProviderItems]);

  // Memoize menu items to avoid creating new references every render
  const menuItems = useMemo(() => {
    const items: JSX.Element[] = [];
    let itemIndex = 0;

    if (groupedItems.common.length > 0) {
      items.push(
        <TypeaheadComboboxSectionHeader key="common-header">
          {intl.formatMessage({
            defaultMessage: 'Common Providers',
            description: 'Section header for common providers',
          })}
        </TypeaheadComboboxSectionHeader>,
      );

      groupedItems.common.forEach((item) => {
        items.push(
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
      items.push(
        <TypeaheadComboboxSectionHeader key="other-header">
          {intl.formatMessage({
            defaultMessage: 'Other Providers',
            description: 'Section header for other providers',
          })}
        </TypeaheadComboboxSectionHeader>,
      );

      groupedItems.other.forEach((item) => {
        items.push(
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

    return items;
  }, [groupedItems, comboboxState, intl, componentIdPrefix]);

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
