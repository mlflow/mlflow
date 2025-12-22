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
import { useMemo, useCallback, useState } from 'react';

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

interface ProviderSelectComboboxProps {
  providerItems: ProviderItem[];
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix: string;
}

const ProviderSelectCombobox = ({
  providerItems,
  value,
  onChange,
  disabled,
  error,
  componentIdPrefix,
}: ProviderSelectComboboxProps) => {
  const intl = useIntl();
  const [filteredItems, setFilteredItems] = useState<(ProviderItem | null)[]>(providerItems);

  const selectedItem = useMemo(() => {
    return providerItems.find((item) => item.provider === value) ?? null;
  }, [providerItems, value]);

  const handleChange = useCallback(
    (item: ProviderItem | null) => {
      onChange(item?.provider ?? '');
    },
    [onChange],
  );

  const deferredFormOnChange = useCallback(
    (item: ProviderItem | null) => {
      setTimeout(() => handleChange(item), 0);
    },
    [handleChange],
  );

  const comboboxState = useComboboxState<ProviderItem | null>({
    componentId: componentIdPrefix,
    allItems: providerItems,
    items: filteredItems,
    setItems: setFilteredItems,
    multiSelect: false,
    itemToString: (item) => item?.displayName ?? '',
    matcher: (item, query) => {
      if (!item) return false;
      const lowerQuery = query.toLowerCase();
      return item.displayName.toLowerCase().includes(lowerQuery) || item.provider.toLowerCase().includes(lowerQuery);
    },
    formValue: selectedItem,
    formOnChange: deferredFormOnChange,
    initialInputValue: selectedItem?.displayName ?? '',
  });

  const handleFocus = useCallback(() => {
    if (selectedItem) {
      comboboxState.setInputValue('');
      onChange('');
    }
  }, [selectedItem, comboboxState, onChange]);

  const groupedItems = useMemo(() => {
    const validItems = filteredItems.filter((item): item is ProviderItem => item !== null);
    const common = validItems.filter((item) => item.group === 'common');
    const other = validItems.filter((item) => item.group === 'other');
    return { common, other };
  }, [filteredItems]);

  const menuItems = useMemo(() => {
    const commonOffset = 0;
    const otherOffset = groupedItems.common.length;

    return [
      ...(groupedItems.common.length > 0
        ? [
            <TypeaheadComboboxSectionHeader key="common-header">
              {intl.formatMessage({
                defaultMessage: 'Common Providers',
                description: 'Section header for common providers',
              })}
            </TypeaheadComboboxSectionHeader>,
            ...groupedItems.common.map((item, idx) => (
              <TypeaheadComboboxMenuItem
                key={item.provider}
                item={item}
                index={commonOffset + idx}
                comboboxState={comboboxState}
                data-testid={`${componentIdPrefix}.option.${item.provider}`}
              >
                {item.displayName}
              </TypeaheadComboboxMenuItem>
            )),
          ]
        : []),
      ...(groupedItems.other.length > 0
        ? [
            <TypeaheadComboboxSectionHeader key="other-header">
              {intl.formatMessage({
                defaultMessage: 'Other Providers',
                description: 'Section header for other providers',
              })}
            </TypeaheadComboboxSectionHeader>,
            ...groupedItems.other.map((item, idx) => (
              <TypeaheadComboboxMenuItem
                key={item.provider}
                item={item}
                index={otherOffset + idx}
                comboboxState={comboboxState}
                data-testid={`${componentIdPrefix}.option.${item.provider}`}
              >
                {item.displayName}
              </TypeaheadComboboxMenuItem>
            )),
          ]
        : []),
    ];
  }, [groupedItems, comboboxState, intl, componentIdPrefix]);

  return (
    <>
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
          onFocus={handleFocus}
        />
        <TypeaheadComboboxMenu comboboxState={comboboxState} matchTriggerWidth minWidth={300}>
          {menuItems}
        </TypeaheadComboboxMenu>
      </TypeaheadComboboxRoot>
      {error && <FormUI.Message type="error" message={error} />}
    </>
  );
};

export const ProviderSelect = ({
  value,
  onChange,
  disabled,
  error,
  componentIdPrefix = 'mlflow.gateway.provider-select',
}: ProviderSelectProps) => {
  const { theme } = useDesignSystemTheme();
  const { data: providers, isLoading } = useProvidersQuery();

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

  if (isLoading || allProviderItems.length === 0) {
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
      <ProviderSelectCombobox
        providerItems={allProviderItems}
        value={value}
        onChange={onChange}
        disabled={disabled}
        error={error}
        componentIdPrefix={componentIdPrefix}
      />
    </div>
  );
};
