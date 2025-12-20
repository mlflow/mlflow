import { useMemo } from 'react';
import { FormUI, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useProvidersQuery } from '../../hooks/useProvidersQuery';
import { formatProviderName, buildProviderGroups, COMMON_PROVIDERS } from '../../utils/providerUtils';
import {
  NavigableCombobox,
  createItem,
  createGroup,
  createTwoTierConfig,
  type ComboboxSelectableItem,
  type ComboboxGroupItem,
  type NavigableComboboxConfig,
} from '../../../common/components/navigable-combobox';

interface NavigableProviderSelectProps {
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix?: string;
}

export const NavigableProviderSelect = ({
  value,
  onChange,
  disabled,
  error,
  componentIdPrefix = 'mlflow.gateway.provider-select',
}: NavigableProviderSelectProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { data: providers, isLoading, error: queryError } = useProvidersQuery();

  const config = useMemo((): NavigableComboboxConfig<string> | null => {
    if (!providers) return null;

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

    const commonItems: (ComboboxSelectableItem<string> | ComboboxGroupItem<string>)[] = [];

    for (const group of groups) {
      const isCommonGroup = group.groupId === 'openai_azure' || group.groupId === 'vertex_ai';
      if (isCommonGroup) {
        commonItems.push(
          createGroup(
            group.groupId,
            group.displayName,
            group.providers.map((p) => createItem(p, formatProviderName(p), p)),
          ),
        );
      }
    }

    for (const provider of commonUngrouped) {
      commonItems.push(createItem(provider, formatProviderName(provider), provider));
    }

    const getCommonProviderIndex = (item: ComboboxSelectableItem<string> | ComboboxGroupItem<string>): number => {
      const provider = item.type === 'group' ? item.children[0]?.value : item.value;
      const index = COMMON_PROVIDERS.indexOf(provider as typeof COMMON_PROVIDERS[number]);
      return index === -1 ? Infinity : index;
    };
    commonItems.sort((a, b) => getCommonProviderIndex(a) - getCommonProviderIndex(b));

    otherUngrouped.sort((a, b) => formatProviderName(a).localeCompare(formatProviderName(b)));

    const litellmItems: ComboboxSelectableItem<string>[] = otherUngrouped.map((provider) =>
      createItem(provider, formatProviderName(provider), provider),
    );

    return createTwoTierConfig({
      mainViewId: 'common',
      mainItems: commonItems,
      moreViewId: 'litellm',
      moreViewLabel: intl.formatMessage(
        {
          defaultMessage: 'LiteLLM ({count} providers)',
          description: 'Navigation to LiteLLM providers',
        },
        { count: litellmItems.length },
      ),
      moreItems: litellmItems,
      backLabel: intl.formatMessage({
        defaultMessage: 'Back to common providers',
        description: 'Navigation back to common providers list',
      }),
    });
  }, [providers, intl]);

  if (queryError) {
    return (
      <div>
        <FormUI.Label htmlFor={componentIdPrefix}>
          <FormattedMessage defaultMessage="Provider" description="Label for provider select field" />
        </FormUI.Label>
        <FormUI.Message type="error" message={queryError.message || 'Failed to load providers'} />
      </div>
    );
  }

  if (isLoading || !config) {
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
      <NavigableCombobox
        componentId={componentIdPrefix}
        config={config}
        value={value || null}
        onChange={(newValue) => onChange(newValue ?? '')}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search for a provider...',
          description: 'Placeholder for provider search input',
        })}
        disabled={disabled}
        error={error}
        minMenuWidth={300}
        clearOnOpen
        showToggleButton
      />
    </div>
  );
};
