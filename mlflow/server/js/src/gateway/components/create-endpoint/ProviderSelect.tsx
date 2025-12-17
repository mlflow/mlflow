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
  SimpleSelect,
  SimpleSelectOption,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useProvidersQuery } from '../../hooks/useProvidersQuery';
import {
  groupProviders,
  formatProviderName,
  buildProviderGroups,
  getProviderGroupId,
  type ProviderGroup,
  COMMON_PROVIDERS,
} from '../../utils/providerUtils';
import { useMemo, useCallback, useState } from 'react';

interface ProviderSelectProps {
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix?: string;
}

type SelectableItem = ProviderGroupItem | ProviderItem;

interface ProviderGroupItem {
  type: 'group';
  groupId: string;
  displayName: string;
  defaultProvider: string;
  providers: string[];
  category: 'common' | 'other';
}

interface ProviderItem {
  type: 'provider';
  provider: string;
  displayName: string;
  category: 'common' | 'other';
}

interface ProviderSelectComboboxProps {
  selectableItems: SelectableItem[];
  selectedGroupId: string | null;
  onSelectGroup: (groupId: string | null, defaultProvider?: string) => void;
  onSelectProvider: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix: string;
}

const ProviderSelectCombobox = ({
  selectableItems,
  selectedGroupId,
  onSelectGroup,
  onSelectProvider,
  disabled,
  error,
  componentIdPrefix,
}: ProviderSelectComboboxProps) => {
  const intl = useIntl();
  const [filteredItems, setFilteredItems] = useState<(SelectableItem | null)[]>(selectableItems);

  const selectedItem = useMemo(() => {
    if (!selectedGroupId) return null;
    return (
      selectableItems.find((item) => {
        if (item.type === 'group') return item.groupId === selectedGroupId;
        return item.provider === selectedGroupId;
      }) ?? null
    );
  }, [selectableItems, selectedGroupId]);

  const handleChange = useCallback(
    (item: SelectableItem | null) => {
      if (!item) {
        onSelectGroup(null);
        return;
      }
      if (item.type === 'group') {
        onSelectGroup(item.groupId, item.defaultProvider);
      } else {
        onSelectProvider(item.provider);
      }
    },
    [onSelectGroup, onSelectProvider],
  );

  const deferredFormOnChange = useCallback(
    (item: SelectableItem | null) => {
      setTimeout(() => handleChange(item), 0);
    },
    [handleChange],
  );

  const comboboxState = useComboboxState<SelectableItem | null>({
    componentId: componentIdPrefix,
    allItems: selectableItems,
    items: filteredItems,
    setItems: setFilteredItems,
    multiSelect: false,
    itemToString: (item) => item?.displayName ?? '',
    matcher: (item, query) => {
      if (!item) return false;
      const lowerQuery = query.toLowerCase();
      return item.displayName.toLowerCase().includes(lowerQuery);
    },
    formValue: selectedItem,
    formOnChange: deferredFormOnChange,
    initialInputValue: selectedItem?.displayName ?? '',
  });

  const handleFocus = useCallback(() => {
    if (selectedItem) {
      comboboxState.setInputValue('');
      onSelectGroup(null);
    }
  }, [selectedItem, comboboxState, onSelectGroup]);

  const groupedItems = useMemo(() => {
    const validItems = filteredItems.filter((item): item is SelectableItem => item !== null);
    const common = validItems.filter((item) => item.category === 'common');
    const other = validItems.filter((item) => item.category === 'other');
    return { common, other };
  }, [filteredItems]);

  const getItemKey = (item: SelectableItem) => {
    return item.type === 'group' ? `group-${item.groupId}` : `provider-${item.provider}`;
  };

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
            key={getItemKey(item)}
            item={item}
            index={itemIndex}
            comboboxState={comboboxState}
            data-testid={`${componentIdPrefix}.option.${getItemKey(item)}`}
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
            key={getItemKey(item)}
            item={item}
            index={itemIndex}
            comboboxState={comboboxState}
            data-testid={`${componentIdPrefix}.option.${getItemKey(item)}`}
          >
            {item.displayName}
          </TypeaheadComboboxMenuItem>,
        );
        itemIndex++;
      });
    }

    return items;
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

interface VariantSelectProps {
  group: ProviderGroup;
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  componentIdPrefix: string;
}

const VariantSelect = ({ group, value, onChange, disabled, componentIdPrefix }: VariantSelectProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const variantLabel =
    group.groupId === 'openai_azure'
      ? intl.formatMessage({
          defaultMessage: 'Platform',
          description: 'Label for OpenAI/Azure platform selector',
        })
      : intl.formatMessage({
          defaultMessage: 'Variant',
          description: 'Label for provider variant selector',
        });

  return (
    <div css={{ marginTop: theme.spacing.md }}>
      <FormUI.Label htmlFor={`${componentIdPrefix}.variant`}>{variantLabel}</FormUI.Label>
      <SimpleSelect
        componentId={`${componentIdPrefix}.variant`}
        id={`${componentIdPrefix}.variant`}
        value={value}
        onChange={({ target }) => onChange(target.value)}
        disabled={disabled}
        css={{ width: '100%' }}
      >
        {group.providers.map((provider) => (
          <SimpleSelectOption key={provider} value={provider}>
            {formatProviderName(provider)}
          </SimpleSelectOption>
        ))}
      </SimpleSelect>
    </div>
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

  const { selectableItems, providerGroupsMap } = useMemo(() => {
    if (!providers) return { selectableItems: [], providerGroupsMap: new Map<string, ProviderGroup>() };

    const { groups, ungroupedProviders } = buildProviderGroups(providers);
    const { common: commonUngrouped, other: otherUngrouped } = groupProviders(ungroupedProviders);

    const items: SelectableItem[] = [];
    const groupsMap = new Map<string, ProviderGroup>();

    for (const group of groups) {
      groupsMap.set(group.groupId, group);
      const isCommon = group.groupId === 'openai_azure' || group.groupId === 'vertex_ai';
      items.push({
        type: 'group',
        groupId: group.groupId,
        displayName: group.displayName,
        defaultProvider: group.defaultProvider,
        providers: group.providers,
        category: isCommon ? 'common' : 'other',
      });
    }

    for (const provider of commonUngrouped) {
      items.push({
        type: 'provider',
        provider,
        displayName: formatProviderName(provider),
        category: 'common',
      });
    }

    for (const provider of otherUngrouped) {
      items.push({
        type: 'provider',
        provider,
        displayName: formatProviderName(provider),
        category: 'other',
      });
    }

    const getCommonProviderIndex = (item: SelectableItem): number => {
      const provider = item.type === 'group' ? item.defaultProvider : item.provider;
      const index = COMMON_PROVIDERS.indexOf(provider as typeof COMMON_PROVIDERS[number]);
      return index === -1 ? Infinity : index;
    };

    items.sort((a, b) => {
      if (a.category !== b.category) {
        return a.category === 'common' ? -1 : 1;
      }
      if (a.category === 'common') {
        return getCommonProviderIndex(a) - getCommonProviderIndex(b);
      }
      return a.displayName.localeCompare(b.displayName);
    });

    return { selectableItems: items, providerGroupsMap: groupsMap };
  }, [providers]);

  const selectedGroupId = useMemo(() => {
    if (!value) return null;
    const groupId = getProviderGroupId(value);
    if (groupId && providerGroupsMap.has(groupId)) {
      return groupId;
    }
    const isUngroupedProvider = selectableItems.some((item) => item.type === 'provider' && item.provider === value);
    return isUngroupedProvider ? value : null;
  }, [value, providerGroupsMap, selectableItems]);

  const activeGroup = useMemo(() => {
    if (!selectedGroupId) return null;
    return providerGroupsMap.get(selectedGroupId) ?? null;
  }, [selectedGroupId, providerGroupsMap]);

  const handleSelectGroup = useCallback(
    (groupId: string | null, defaultProvider?: string) => {
      if (!groupId) {
        onChange('');
        return;
      }
      if (defaultProvider) {
        onChange(defaultProvider);
      }
    },
    [onChange],
  );

  const handleSelectProvider = useCallback(
    (provider: string) => {
      onChange(provider);
    },
    [onChange],
  );

  const handleVariantChange = useCallback(
    (provider: string) => {
      onChange(provider);
    },
    [onChange],
  );

  if (isLoading || selectableItems.length === 0) {
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
        selectableItems={selectableItems}
        selectedGroupId={selectedGroupId}
        onSelectGroup={handleSelectGroup}
        onSelectProvider={handleSelectProvider}
        disabled={disabled}
        error={error}
        componentIdPrefix={componentIdPrefix}
      />
      {activeGroup && (
        <VariantSelect
          group={activeGroup}
          value={value}
          onChange={handleVariantChange}
          disabled={disabled}
          componentIdPrefix={componentIdPrefix}
        />
      )}
    </div>
  );
};
