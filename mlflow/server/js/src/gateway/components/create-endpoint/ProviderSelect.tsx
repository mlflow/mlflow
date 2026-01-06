import {
  TypeaheadComboboxRoot,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  useComboboxState,
  useDesignSystemTheme,
  FormUI,
  Spinner,
  ChevronLeftIcon,
  ChevronRightIcon,
  Alert,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useProvidersQuery } from '../../hooks/useProvidersQuery';
import { formatProviderName, buildProviderGroups, COMMON_PROVIDERS } from '../../utils/providerUtils';
import { useMemo, useCallback, useState, useEffect, useRef } from 'react';

interface ProviderSelectProps {
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix?: string;
}

type MenuItem = ProviderGroupItem | ProviderItem | OthersNavigationItem | BackNavigationItem;

interface ProviderGroupItem {
  type: 'group';
  groupId: string;
  displayName: string;
  defaultProvider: string;
  providers: string[];
}

interface ProviderItem {
  type: 'provider';
  provider: string;
  displayName: string;
  isLiteLLM?: boolean;
}

interface OthersNavigationItem {
  type: 'others-nav';
  displayName: string;
  count: number;
}

interface BackNavigationItem {
  type: 'back-nav';
  displayName: string;
  returnTo: 'common' | 'litellm';
}

interface GroupViewMode {
  type: 'group';
  group: ProviderGroupItem;
}

type ViewMode = 'common' | 'litellm' | GroupViewMode;

interface ProviderSelectComboboxProps {
  commonItems: MenuItem[];
  litellmItems: MenuItem[];
  otherProvidersCount: number;
  selectedProvider: string;
  onSelectProvider: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix: string;
}

const ProviderSelectCombobox = ({
  commonItems,
  litellmItems,
  otherProvidersCount,
  selectedProvider,
  onSelectProvider,
  disabled,
  error,
  componentIdPrefix,
}: ProviderSelectComboboxProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [viewMode, setViewMode] = useState<ViewMode>('common');
  const [searchQuery, setSearchQuery] = useState('');

  const othersNavItem: OthersNavigationItem = useMemo(
    () => ({
      type: 'others-nav',
      displayName: intl.formatMessage(
        {
          defaultMessage: 'LiteLLM ({count} providers)',
          description: 'Navigation to LiteLLM providers',
        },
        { count: otherProvidersCount },
      ),
      count: otherProvidersCount,
    }),
    [intl, otherProvidersCount],
  );

  const backToCommonNavItem: BackNavigationItem = useMemo(
    () => ({
      type: 'back-nav',
      displayName: intl.formatMessage({
        defaultMessage: 'Back to common providers',
        description: 'Navigation back to common providers list',
      }),
      returnTo: 'common',
    }),
    [intl],
  );

  const displayItems = useMemo((): MenuItem[] => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();

      const matchingCommon: MenuItem[] = [];
      for (const item of commonItems) {
        if (item.type === 'others-nav' || item.type === 'back-nav') continue;

        if (item.type === 'group') {
          const matchingProviders = item.providers.filter((p) => formatProviderName(p).toLowerCase().includes(query));
          for (const provider of matchingProviders) {
            matchingCommon.push({
              type: 'provider',
              provider,
              displayName: formatProviderName(provider),
            });
          }
        } else if (item.displayName.toLowerCase().includes(query)) {
          matchingCommon.push(item);
        }
      }

      const matchingLiteLLM = litellmItems.filter((item) => {
        if (item.type === 'others-nav' || item.type === 'back-nav') return false;
        return item.displayName.toLowerCase().includes(query);
      });

      if (matchingLiteLLM.length > 0 && matchingCommon.length === 0) {
        return [backToCommonNavItem, ...matchingLiteLLM];
      } else if (matchingLiteLLM.length > 0) {
        const litellmSection: OthersNavigationItem = {
          type: 'others-nav',
          displayName: intl.formatMessage(
            {
              defaultMessage: 'LiteLLM ({count} matches)',
              description: 'Navigation showing matching LiteLLM providers',
            },
            { count: matchingLiteLLM.length },
          ),
          count: matchingLiteLLM.length,
        };
        return [...matchingCommon, litellmSection, ...matchingLiteLLM];
      }
      return matchingCommon;
    }

    if (viewMode === 'litellm') {
      return [backToCommonNavItem, ...litellmItems];
    }
    if (typeof viewMode === 'object' && viewMode.type === 'group') {
      const groupProviderItems: ProviderItem[] = viewMode.group.providers.map((provider) => ({
        type: 'provider',
        provider,
        displayName: formatProviderName(provider),
      }));
      return [backToCommonNavItem, ...groupProviderItems];
    }
    return otherProvidersCount > 0 ? [...commonItems, othersNavItem] : commonItems;
  }, [searchQuery, viewMode, commonItems, litellmItems, othersNavItem, backToCommonNavItem, otherProvidersCount, intl]);

  const selectedDisplayName = useMemo(() => {
    if (!selectedProvider) return '';
    for (const item of [...commonItems, ...litellmItems]) {
      if (item.type === 'provider' && item.provider === selectedProvider) {
        return item.displayName;
      }
      if (item.type === 'group' && item.providers.includes(selectedProvider)) {
        return formatProviderName(selectedProvider);
      }
    }
    return formatProviderName(selectedProvider);
  }, [selectedProvider, commonItems, litellmItems]);

  const allItems = useMemo(() => [...commonItems, ...litellmItems], [commonItems, litellmItems]);

  const setItemsNoop = useCallback(() => {}, []);

  const handleInputValueChange = useCallback((val: React.SetStateAction<string>) => {
    if (typeof val === 'string') {
      setSearchQuery(val.toLowerCase());
    }
  }, []);

  const customMatcher = useCallback((): boolean => {
    return true;
  }, []);

  const handleMenuSelection = useCallback(
    (item: MenuItem | null) => {
      if (!item) {
        onSelectProvider('');
        setSearchQuery('');
        setViewMode('common');
        return;
      }

      if (item.type === 'others-nav' || item.type === 'back-nav' || item.type === 'group') {
        return;
      }

      onSelectProvider(item.provider);
      setSearchQuery('');
      setViewMode('common');
    },
    [onSelectProvider],
  );

  const comboboxState = useComboboxState<MenuItem | null>({
    componentId: componentIdPrefix,
    allItems: allItems,
    items: displayItems,
    setItems: setItemsNoop,
    setInputValue: handleInputValueChange,
    multiSelect: false,
    itemToString: (item) => {
      if (!item) return '';
      return item.displayName;
    },
    matcher: customMatcher,
    formValue: null,
    formOnChange: handleMenuSelection,
    initialInputValue: selectedDisplayName,
  });

  const prevIsOpenRef = useRef(comboboxState.isOpen);
  useEffect(() => {
    const wasOpen = prevIsOpenRef.current;
    const isOpen = comboboxState.isOpen;
    prevIsOpenRef.current = isOpen;

    if (isOpen && !wasOpen) {
      setSearchQuery('');
      setViewMode('common');
      comboboxState.setInputValue('');
    }
  }, [comboboxState.isOpen, comboboxState.setInputValue]);

  useEffect(() => {
    if (!comboboxState.isOpen) {
      comboboxState.setInputValue(selectedDisplayName);
    }
  }, [selectedDisplayName, comboboxState.isOpen, comboboxState.setInputValue]);

  const getItemKey = (item: MenuItem) => {
    if (item.type === 'group') return `group-${item.groupId}`;
    if (item.type === 'others-nav') return 'others-nav';
    if (item.type === 'back-nav') return 'back-nav';
    return `provider-${item.provider}`;
  };

  return (
    <>
      <TypeaheadComboboxRoot id={componentIdPrefix} comboboxState={comboboxState}>
        <TypeaheadComboboxInput
          placeholder={intl.formatMessage({
            defaultMessage: 'Search for a provider...',
            description: 'Placeholder for provider search input',
          })}
          comboboxState={comboboxState}
          validationState={error ? 'error' : undefined}
          disabled={disabled}
          showComboboxToggleButton
          allowClear={false}
        />
        <TypeaheadComboboxMenu comboboxState={comboboxState} matchTriggerWidth minWidth={300}>
          {displayItems.map((item, index) => {
            if (item.type === 'others-nav') {
              return (
                <TypeaheadComboboxMenuItem
                  key={getItemKey(item)}
                  item={item}
                  index={index}
                  comboboxState={comboboxState}
                  data-testid={`${componentIdPrefix}.option.others-nav`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setViewMode('litellm');
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                      color: theme.colors.textSecondary,
                    }}
                  >
                    <span>{item.displayName}</span>
                    <ChevronRightIcon css={{ marginLeft: theme.spacing.sm }} />
                  </div>
                </TypeaheadComboboxMenuItem>
              );
            }

            if (item.type === 'back-nav') {
              return (
                <TypeaheadComboboxMenuItem
                  key={getItemKey(item)}
                  item={item}
                  index={index}
                  comboboxState={comboboxState}
                  data-testid={`${componentIdPrefix}.option.back-nav`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setViewMode(item.returnTo);
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      color: theme.colors.textSecondary,
                      borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                      paddingBottom: theme.spacing.xs,
                      marginBottom: theme.spacing.xs,
                    }}
                  >
                    <ChevronLeftIcon css={{ marginRight: theme.spacing.sm }} />
                    <span>{item.displayName}</span>
                  </div>
                </TypeaheadComboboxMenuItem>
              );
            }

            if (item.type === 'group') {
              return (
                <TypeaheadComboboxMenuItem
                  key={getItemKey(item)}
                  item={item}
                  index={index}
                  comboboxState={comboboxState}
                  data-testid={`${componentIdPrefix}.option.${getItemKey(item)}`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setViewMode({ type: 'group', group: item });
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      width: '100%',
                    }}
                  >
                    <span>{item.displayName}</span>
                    <ChevronRightIcon css={{ marginLeft: theme.spacing.sm, color: theme.colors.textSecondary }} />
                  </div>
                </TypeaheadComboboxMenuItem>
              );
            }

            return (
              <TypeaheadComboboxMenuItem
                key={getItemKey(item)}
                item={item}
                index={index}
                comboboxState={comboboxState}
                data-testid={`${componentIdPrefix}.option.${getItemKey(item)}`}
              >
                {item.displayName}
              </TypeaheadComboboxMenuItem>
            );
          })}
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
  const { data: providers, isLoading, error: queryError } = useProvidersQuery();

  const { commonItems, litellmItems, otherProviders } = useMemo(() => {
    if (!providers)
      return {
        commonItems: [] as MenuItem[],
        litellmItems: [] as MenuItem[],
        otherProviders: [] as string[],
      };

    const { groups, ungroupedProviders } = buildProviderGroups(providers);
    const commonSet = new Set<string>(COMMON_PROVIDERS);

    const commonUngrouped: string[] = [];
    const otherUngrouped: string[] = [];
    for (const provider of ungroupedProviders) {
      if (commonSet.has(provider)) {
        commonUngrouped.push(provider);
      } else {
        otherUngrouped.push(provider);
      }
    }

    const common: MenuItem[] = [];

    for (const group of groups) {
      const isCommonGroup = group.groupId === 'openai_azure' || group.groupId === 'vertex_ai';
      if (isCommonGroup) {
        common.push({
          type: 'group',
          groupId: group.groupId,
          displayName: group.displayName,
          defaultProvider: group.defaultProvider,
          providers: group.providers,
        });
      }
    }

    for (const provider of commonUngrouped) {
      common.push({
        type: 'provider',
        provider,
        displayName: formatProviderName(provider),
      });
    }

    const getCommonProviderIndex = (item: MenuItem): number => {
      if (item.type === 'others-nav' || item.type === 'back-nav') return Infinity;
      const provider = item.type === 'group' ? item.defaultProvider : item.provider;
      const index = COMMON_PROVIDERS.indexOf(provider as (typeof COMMON_PROVIDERS)[number]);
      return index === -1 ? Infinity : index;
    };
    common.sort((a, b) => getCommonProviderIndex(a) - getCommonProviderIndex(b));

    otherUngrouped.sort((a, b) => formatProviderName(a).localeCompare(formatProviderName(b)));

    const litellm: MenuItem[] = otherUngrouped.map((provider) => ({
      type: 'provider',
      provider,
      displayName: formatProviderName(provider),
      isLiteLLM: true,
    }));

    return {
      commonItems: common,
      litellmItems: litellm,
      otherProviders: otherUngrouped,
    };
  }, [providers]);

  const handleSelectProvider = useCallback(
    (provider: string) => {
      onChange(provider);
    },
    [onChange],
  );

  if (queryError) {
    return (
      <div>
        <FormUI.Label htmlFor={componentIdPrefix}>
          <FormattedMessage defaultMessage="Provider" description="Label for provider select field" />
        </FormUI.Label>
        <Alert
          componentId={`${componentIdPrefix}.error`}
          type="error"
          message={queryError.message}
          css={{ marginTop: theme.spacing.xs }}
        />
      </div>
    );
  }

  if (isLoading || commonItems.length === 0) {
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
        commonItems={commonItems}
        litellmItems={litellmItems}
        otherProvidersCount={otherProviders.length}
        selectedProvider={value}
        onSelectProvider={handleSelectProvider}
        disabled={disabled}
        error={error}
        componentIdPrefix={componentIdPrefix}
      />
    </div>
  );
};
