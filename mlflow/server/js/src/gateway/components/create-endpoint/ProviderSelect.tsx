import { useMemo, useCallback } from 'react';
import { useDesignSystemTheme, FormUI, Spinner, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { NavigableCombobox } from '../../../common/components/navigable-combobox/NavigableCombobox';
import type {
  NavigableComboboxConfig,
  ComboboxModalTriggerItem,
  ComboboxGroupItem,
  ComboboxSelectableItem,
} from '../../../common/components/navigable-combobox/types';
import type { SelectorItem } from '../../../common/components/selector-modal/types';
import { useProvidersQuery } from '../../hooks/useProvidersQuery';
import { formatProviderName, buildProviderGroups, COMMON_PROVIDERS } from '../../utils/providerUtils';

interface ProviderSelectProps {
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix?: string;
}

export const ProviderSelect = ({
  value,
  onChange,
  disabled,
  error,
  componentIdPrefix = 'mlflow.gateway.provider-select',
}: ProviderSelectProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { data: providers, isLoading, error: queryError } = useProvidersQuery();

  const { config, hasOtherProviders } = useMemo((): {
    config: NavigableComboboxConfig<string>;
    hasOtherProviders: boolean;
  } => {
    if (!providers) {
      return {
        config: { initialViewId: 'main', views: [{ id: 'main', items: [] }] },
        hasOtherProviders: false,
      };
    }

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

    const items: (ComboboxSelectableItem<string> | ComboboxGroupItem<string> | ComboboxModalTriggerItem<string>)[] = [];

    for (const group of groups) {
      const isCommonGroup = group.groupId === 'openai_azure' || group.groupId === 'vertex_ai';
      if (isCommonGroup) {
        const groupItem: ComboboxGroupItem<string> = {
          type: 'group',
          key: `group-${group.groupId}`,
          label: group.displayName,
          backLabel: intl.formatMessage({
            defaultMessage: 'Back to providers',
            description: 'Navigation back to main provider list',
          }),
          children: group.providers.map((provider) => ({
            type: 'item' as const,
            key: provider,
            label: formatProviderName(provider),
            value: provider,
          })),
        };
        items.push(groupItem);
      }
    }

    for (const provider of commonUngrouped) {
      items.push({
        type: 'item',
        key: provider,
        label: formatProviderName(provider),
        value: provider,
      });
    }

    const getCommonProviderIndex = (
      item: ComboboxSelectableItem<string> | ComboboxGroupItem<string> | ComboboxModalTriggerItem<string>,
    ): number => {
      if (item.type === 'modal-trigger') return Infinity;
      const providerKey = item.type === 'group' ? item.children[0]?.value : item.value;
      const index = COMMON_PROVIDERS.indexOf(providerKey as (typeof COMMON_PROVIDERS)[number]);
      return index === -1 ? Infinity : index;
    };
    items.sort((a, b) => getCommonProviderIndex(a) - getCommonProviderIndex(b));

    if (otherUngrouped.length > 0) {
      otherUngrouped.sort((a, b) => formatProviderName(a).localeCompare(formatProviderName(b)));

      const modalItems: SelectorItem<string>[] = otherUngrouped.map((provider) => ({
        key: provider,
        label: formatProviderName(provider),
        value: provider,
      }));

      const modalTrigger: ComboboxModalTriggerItem<string> = {
        type: 'modal-trigger',
        key: 'others',
        label: intl.formatMessage(
          {
            defaultMessage: 'LiteLLM ({count} providers)',
            description: 'Link to open modal with all LiteLLM providers',
          },
          { count: otherUngrouped.length },
        ),
        modalTitle: intl.formatMessage({
          defaultMessage: 'Select a Provider',
          description: 'Modal title for provider selection',
        }),
        modalSearchPlaceholder: intl.formatMessage({
          defaultMessage: 'Search providers...',
          description: 'Search placeholder in provider modal',
        }),
        modalItems,
      };
      items.push(modalTrigger);
    }

    return {
      config: {
        initialViewId: 'main',
        views: [{ id: 'main', items }],
      },
      hasOtherProviders: otherUngrouped.length > 0,
    };
  }, [providers, intl]);

  const handleChange = useCallback(
    (newValue: string | null) => {
      onChange(newValue ?? '');
    },
    [onChange],
  );

  const renderModalItem = useCallback(
    (item: SelectorItem<string>, defaultContent: React.ReactNode) => (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '100%',
        }}
      >
        {defaultContent}
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          LiteLLM
        </Typography.Text>
      </div>
    ),
    [theme.typography.fontSizeSm],
  );

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

  if (isLoading || config.views[0].items.length === 0) {
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
        onChange={handleChange}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search for a provider...',
          description: 'Placeholder for provider search input',
        })}
        disabled={disabled}
        error={error}
        minMenuWidth={300}
        renderModalItem={renderModalItem}
      />
    </div>
  );
};
