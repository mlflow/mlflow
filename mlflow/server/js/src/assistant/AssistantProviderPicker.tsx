import {
  CheckIcon,
  ChevronDownIcon,
  DropdownMenu,
  PlusIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { GATEWAY_PROVIDER_ID } from './constants';
import { getAssistantProvider, getLlmProviderDisplay } from './providerRegistry';
import type { AssistantProviderSelection, ProviderInfo, ResolvedProviderInfo } from './types';
import { useEndpointsQuery } from '../gateway/hooks/useEndpointsQuery';
import { GatewayRoutePaths } from '../gateway/routes';
import type { Endpoint } from '../gateway/types';

const GATEWAY_VENDOR_ORDER = ['openai', 'anthropic', 'gemini'];
const GATEWAY_VENDOR_MENU_LABELS = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  gemini: 'Gemini',
} satisfies Record<string, string>;
const LOCAL_PROVIDER_MENU_LABELS = {
  claude_code: 'Claude Code',
  codex: 'Codex CLI',
  ollama: 'Ollama',
} satisfies Record<string, string>;
const PROVIDER_SELECTION_KIND = {
  Provider: 'provider',
  Gateway: 'gateway',
} as const;
type OnSelectProvider = (selection: AssistantProviderSelection) => void;

const gatewayVendorEndpointName = (vendor: string): string => `mlflow-assistant-${vendor}`;

const orderedGatewayVendors = (options: Record<string, string[]>): string[] => {
  const known = GATEWAY_VENDOR_ORDER.filter((vendor) => vendor in options);
  const rest = Object.keys(options)
    .filter((vendor) => !known.includes(vendor))
    .sort();
  return [...known, ...rest];
};

const endpointModelName = (endpoint: Endpoint | undefined): string | undefined =>
  endpoint?.model_mappings?.[0]?.model_definition?.model_name;

const gatewayVendorMenuLabel = (vendor: string): string =>
  (GATEWAY_VENDOR_MENU_LABELS as Record<string, string | undefined>)[vendor] ??
  getLlmProviderDisplay(vendor)?.name ??
  vendor;

const providerMenuLabel = (providerName: string, fallback: string): string =>
  (LOCAL_PROVIDER_MENU_LABELS as Record<string, string | undefined>)[providerName] ?? fallback;

const currentModelOptions = (provider: ResolvedProviderInfo, providers: ProviderInfo[]): string[] => {
  if (provider.model_options?.length) {
    return provider.model_options;
  }
  return providers.find((candidate) => candidate.name === provider.name)?.model_options ?? [];
};

const currentModel = (provider: ResolvedProviderInfo, options: string[]): string => {
  if (provider.model_options?.length) {
    return provider.provider_model ?? options[0] ?? provider.model ?? 'default';
  }
  return provider.model ?? 'default';
};

const GatewayVendorItems = ({
  provider,
  gatewayVendorOptions,
  onSelect,
}: {
  provider: ResolvedProviderInfo;
  gatewayVendorOptions: Record<string, string[]>;
  onSelect: OnSelectProvider;
}) => {
  const { data } = useEndpointsQuery();
  const { theme } = useDesignSystemTheme();
  const endpoints = data ?? [];
  const managedVendors = orderedGatewayVendors(gatewayVendorOptions);
  if (managedVendors.length === 0) {
    return null;
  }

  return (
    <>
      {managedVendors.map((vendor) => {
        const endpointName = gatewayVendorEndpointName(vendor);
        const endpoint = endpoints.find((candidate) => candidate.name === endpointName);
        const modelOptions = gatewayVendorOptions[vendor] ?? [];
        const providerModel = endpointModelName(endpoint) ?? modelOptions[0];
        const vendorDisplay = getLlmProviderDisplay(vendor);
        const label = gatewayVendorMenuLabel(vendor);
        const isSelected = provider.name === GATEWAY_PROVIDER_ID && provider.model_provider === vendor;

        return (
          <DropdownMenu.Item
            componentId="mlflow.assistant.provider_picker.item"
            key={vendor}
            onClick={() =>
              onSelect({
                kind: PROVIDER_SELECTION_KIND.Gateway,
                endpointName,
                gatewayVendor: vendor,
                providerModel,
                modelOptions,
                requiresApiKey: !endpoint,
                hasApiKey: Boolean(endpoint),
              })
            }
          >
            {vendorDisplay?.logo && (
              <img
                src={vendorDisplay.logo}
                alt=""
                aria-hidden
                css={{ width: 16, height: 16, marginRight: theme.spacing.xs, borderRadius: 2 }}
              />
            )}
            <span css={{ flex: 1 }}>{label}</span>
            {isSelected && (
              <CheckIcon
                aria-hidden
                data-testid="assistant-provider-selected-check"
                css={{ marginLeft: theme.spacing.sm }}
              />
            )}
          </DropdownMenu.Item>
        );
      })}
    </>
  );
};

const GatewayEndpointItems = ({
  provider,
  gatewayVendorOptions,
  onSelect,
}: {
  provider: ResolvedProviderInfo;
  gatewayVendorOptions: Record<string, string[]>;
  onSelect: OnSelectProvider;
}) => {
  const { data } = useEndpointsQuery();
  const { theme } = useDesignSystemTheme();
  const endpoints = data ?? [];
  const managedEndpointNames = new Set(Object.keys(gatewayVendorOptions).map(gatewayVendorEndpointName));
  const customEndpoints = endpoints.filter((endpoint) => !managedEndpointNames.has(endpoint.name));

  return (
    <>
      {customEndpoints.map((endpoint) => (
        <DropdownMenu.Item
          componentId="mlflow.assistant.provider_picker.endpoint"
          key={endpoint.name}
          onClick={() => onSelect({ kind: PROVIDER_SELECTION_KIND.Gateway, endpointName: endpoint.name })}
        >
          <span css={{ flex: 1 }}>{endpoint.name}</span>
          {provider.name === GATEWAY_PROVIDER_ID && provider.model === endpoint.name && (
            <CheckIcon
              aria-hidden
              data-testid="assistant-provider-selected-check"
              css={{ marginLeft: theme.spacing.sm }}
            />
          )}
        </DropdownMenu.Item>
      ))}
      <DropdownMenu.Item
        componentId="mlflow.assistant.provider_picker.create_endpoint"
        onClick={() => window.open(`/#${GatewayRoutePaths.createEndpointPage}`, '_blank', 'noopener')}
      >
        <PlusIcon css={{ marginRight: 4 }} />
        <FormattedMessage
          defaultMessage="Create new endpoint"
          description="Option in the assistant provider picker that opens the gateway endpoint creation page"
        />
      </DropdownMenu.Item>
    </>
  );
};

const ProviderMenu = ({
  provider,
  providers,
  gatewayVendorOptions,
  disabled,
  onSelect,
}: {
  provider: ResolvedProviderInfo;
  providers: ProviderInfo[];
  gatewayVendorOptions: Record<string, string[]>;
  disabled?: boolean;
  onSelect: OnSelectProvider;
}) => {
  const { theme } = useDesignSystemTheme();
  const meta = getAssistantProvider(provider.name);
  const llmVendor = provider.model_provider ? getLlmProviderDisplay(provider.model_provider) : undefined;
  const label = provider.model_provider
    ? llmVendor
      ? gatewayVendorMenuLabel(provider.model_provider)
      : (meta?.name ?? provider.name)
    : providerMenuLabel(provider.name, meta?.name ?? provider.name);
  const logo = llmVendor?.logo ?? meta?.logo;
  const ProviderIcon = llmVendor ? undefined : meta?.icon;
  const gatewayProvider = providers.find((candidate) => candidate.name === GATEWAY_PROVIDER_ID);
  const standardProviders = providers.filter((candidate) => candidate.name !== GATEWAY_PROVIDER_ID);
  const hasGatewayVendorOptions = orderedGatewayVendors(gatewayVendorOptions).length > 0;
  const groupLabelStyles = { color: theme.colors.textSecondary };

  const providerItemContent = (candidate: ProviderInfo, candidateLabel: string, isSelected = false) => {
    const candidateMeta = getAssistantProvider(candidate.name);
    const candidateLogo = candidateMeta?.logo;
    const CandidateIcon = candidateMeta?.icon;

    return (
      <>
        {candidateLogo && (
          <img
            src={candidateLogo}
            alt=""
            aria-hidden
            css={{ width: 16, height: 16, marginRight: theme.spacing.xs, borderRadius: 2 }}
          />
        )}
        {!candidateLogo && CandidateIcon && (
          <CandidateIcon
            aria-hidden
            css={{ fontSize: 16, marginRight: theme.spacing.xs, color: theme.colors.textSecondary }}
          />
        )}
        <span css={{ flex: 1 }}>{candidateLabel}</span>
        {isSelected && (
          <CheckIcon
            aria-hidden
            data-testid="assistant-provider-selected-check"
            css={{ marginLeft: theme.spacing.sm }}
          />
        )}
      </>
    );
  };

  const trigger = (
    <button
      type="button"
      disabled={disabled}
      aria-label="Change assistant provider"
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        minWidth: 0,
        border: 'none',
        background: 'transparent',
        padding: 0,
        cursor: disabled ? 'default' : 'pointer',
      }}
    >
      {logo && <img src={logo} alt="" aria-hidden css={{ width: 14, height: 14, flexShrink: 0, borderRadius: 2 }} />}
      {!logo && ProviderIcon && (
        <ProviderIcon aria-hidden css={{ fontSize: 14, flexShrink: 0, color: theme.colors.textSecondary }} />
      )}
      <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
        {label}
      </Typography.Text>
      {!disabled && <ChevronDownIcon css={{ fontSize: 12, color: theme.colors.textSecondary }} />}
    </button>
  );

  if (disabled) {
    return trigger;
  }

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Content side="top" align="start" css={{ minWidth: 220 }}>
        {hasGatewayVendorOptions && (
          <DropdownMenu.Group>
            <DropdownMenu.Label css={groupLabelStyles}>
              <FormattedMessage
                defaultMessage="API providers"
                description="Provider picker section label for assistant providers backed by external hosted APIs"
              />
            </DropdownMenu.Label>
            <GatewayVendorItems provider={provider} gatewayVendorOptions={gatewayVendorOptions} onSelect={onSelect} />
          </DropdownMenu.Group>
        )}
        {hasGatewayVendorOptions && standardProviders.length > 0 && <DropdownMenu.Separator />}
        {standardProviders.length > 0 && (
          <DropdownMenu.Group>
            <DropdownMenu.Label css={groupLabelStyles}>
              <FormattedMessage
                defaultMessage="Local providers"
                description="Provider picker section label for assistant providers that run through a local CLI or local server"
              />
            </DropdownMenu.Label>
            {standardProviders.map((candidate) => {
              const candidateMeta = getAssistantProvider(candidate.name);
              const candidateLabel = providerMenuLabel(candidate.name, candidateMeta?.name ?? candidate.display_name);
              const isSelected = provider.name === candidate.name;

              return (
                <DropdownMenu.Item
                  componentId="mlflow.assistant.provider_picker.item"
                  key={candidate.name}
                  disabled={!candidate.available}
                  onClick={() =>
                    onSelect({
                      kind: PROVIDER_SELECTION_KIND.Provider,
                      name: candidate.name,
                      model: candidate.model_options[0],
                    })
                  }
                >
                  {providerItemContent(candidate, candidateLabel, isSelected)}
                </DropdownMenu.Item>
              );
            })}
          </DropdownMenu.Group>
        )}
        {gatewayProvider && (hasGatewayVendorOptions || standardProviders.length > 0) && <DropdownMenu.Separator />}
        {gatewayProvider && (
          <DropdownMenu.Group>
            <DropdownMenu.Label css={groupLabelStyles}>
              <FormattedMessage
                defaultMessage="Gateway endpoints"
                description="Provider picker section label for custom MLflow AI Gateway endpoints"
              />
            </DropdownMenu.Label>
            <DropdownMenu.Sub>
              <DropdownMenu.SubTrigger>
                {providerItemContent(
                  gatewayProvider,
                  getAssistantProvider(gatewayProvider.name)?.name ?? gatewayProvider.display_name,
                  provider.name === GATEWAY_PROVIDER_ID && !provider.model_provider,
                )}
              </DropdownMenu.SubTrigger>
              <DropdownMenu.SubContent>
                <GatewayEndpointItems
                  provider={provider}
                  gatewayVendorOptions={gatewayVendorOptions}
                  onSelect={onSelect}
                />
              </DropdownMenu.SubContent>
            </DropdownMenu.Sub>
          </DropdownMenu.Group>
        )}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

const ModelPicker = ({
  model,
  options,
  disabled,
  onSelect,
}: {
  model: string;
  options: string[];
  disabled?: boolean;
  onSelect: (model: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  if (options.length === 0) {
    return null;
  }

  const selectedModel = model && model !== 'default' && options.includes(model) ? model : options[0];
  const canChangeModel = !disabled && options.length > 1;
  const trigger = (
    <button
      type="button"
      disabled={!canChangeModel}
      aria-label={canChangeModel ? 'Change assistant model' : 'Assistant model'}
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        minWidth: 0,
        maxWidth: 180,
        border: 'none',
        background: 'transparent',
        padding: 0,
        cursor: canChangeModel ? 'pointer' : 'default',
      }}
    >
      <Typography.Text
        size="sm"
        color="secondary"
        css={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
      >
        {selectedModel}
      </Typography.Text>
      {canChangeModel && <ChevronDownIcon css={{ fontSize: 12, color: theme.colors.textSecondary }} />}
    </button>
  );

  if (!canChangeModel) {
    return trigger;
  }

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Content side="top" align="start" css={{ minWidth: 180 }}>
        {options.map((option) => (
          <DropdownMenu.Item
            componentId="mlflow.assistant.model_picker.item"
            key={option}
            onClick={() => onSelect(option)}
          >
            <span css={{ flex: 1 }}>{option}</span>
            {option === selectedModel && <CheckIcon css={{ marginLeft: theme.spacing.sm }} />}
          </DropdownMenu.Item>
        ))}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

export const AssistantProviderPicker = ({
  provider,
  providers,
  gatewayVendorOptions,
  disabled,
  onSelect,
}: {
  provider: ResolvedProviderInfo;
  providers: ProviderInfo[];
  gatewayVendorOptions: Record<string, string[]>;
  disabled?: boolean;
  onSelect: OnSelectProvider;
}) => {
  const modelOptions = currentModelOptions(provider, providers);

  return (
    <>
      <ProviderMenu
        provider={provider}
        providers={providers}
        gatewayVendorOptions={gatewayVendorOptions}
        disabled={disabled}
        onSelect={onSelect}
      />
      <ModelPicker
        model={currentModel(provider, modelOptions)}
        options={modelOptions}
        disabled={disabled}
        onSelect={(model) => {
          if (provider.name === GATEWAY_PROVIDER_ID && provider.model_provider) {
            onSelect({
              kind: PROVIDER_SELECTION_KIND.Gateway,
              endpointName: provider.model ?? gatewayVendorEndpointName(provider.model_provider),
              gatewayVendor: provider.model_provider,
              providerModel: model,
              modelOptions,
              requiresApiKey: provider.requires_api_key,
              hasApiKey: provider.has_api_key,
            });
          } else {
            onSelect({ kind: PROVIDER_SELECTION_KIND.Provider, name: provider.name, model });
          }
        }}
      />
    </>
  );
};
