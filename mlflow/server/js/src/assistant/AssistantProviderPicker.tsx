import { Fragment } from 'react';
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
  gatewayVendorOptions,
  onSelect,
}: {
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

        return (
          <DropdownMenu.Item
            componentId="mlflow.assistant.provider_picker.item"
            key={vendor}
            onClick={() =>
              onSelect({
                kind: 'gateway',
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
            <span css={{ flex: 1 }}>{vendorDisplay?.name ?? vendor}</span>
          </DropdownMenu.Item>
        );
      })}
      <DropdownMenu.Separator />
    </>
  );
};

const GatewayEndpointItems = ({
  gatewayVendorOptions,
  onSelect,
}: {
  gatewayVendorOptions: Record<string, string[]>;
  onSelect: OnSelectProvider;
}) => {
  const { data } = useEndpointsQuery();
  const endpoints = data ?? [];
  const managedEndpointNames = new Set(Object.keys(gatewayVendorOptions).map(gatewayVendorEndpointName));
  const customEndpoints = endpoints.filter((endpoint) => !managedEndpointNames.has(endpoint.name));

  return (
    <>
      {customEndpoints.map((endpoint) => (
        <DropdownMenu.Item
          componentId="mlflow.assistant.provider_picker.endpoint"
          key={endpoint.name}
          onClick={() => onSelect({ kind: 'gateway', endpointName: endpoint.name })}
        >
          {endpoint.name}
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
  const label = llmVendor?.name ?? meta?.name ?? provider.name;
  const logo = llmVendor?.logo ?? meta?.logo;
  const ProviderIcon = llmVendor ? undefined : meta?.icon;

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
        {providers.map((candidate) => {
          const candidateMeta = getAssistantProvider(candidate.name);
          const candidateLogo = candidateMeta?.logo;
          const CandidateIcon = candidateMeta?.icon;
          const candidateLabel = candidateMeta?.name ?? candidate.display_name;
          const itemContent = (
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
            </>
          );

          if (candidate.name === GATEWAY_PROVIDER_ID) {
            return (
              <Fragment key={candidate.name}>
                <GatewayVendorItems gatewayVendorOptions={gatewayVendorOptions} onSelect={onSelect} />
                <DropdownMenu.Sub>
                  <DropdownMenu.SubTrigger>{itemContent}</DropdownMenu.SubTrigger>
                  <DropdownMenu.SubContent>
                    <GatewayEndpointItems gatewayVendorOptions={gatewayVendorOptions} onSelect={onSelect} />
                  </DropdownMenu.SubContent>
                </DropdownMenu.Sub>
              </Fragment>
            );
          }

          return (
            <DropdownMenu.Item
              componentId="mlflow.assistant.provider_picker.item"
              key={candidate.name}
              disabled={!candidate.available}
              onClick={() => onSelect({ kind: 'provider', name: candidate.name, model: candidate.model_options[0] })}
            >
              {itemContent}
            </DropdownMenu.Item>
          );
        })}
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
  const trigger = (
    <button
      type="button"
      disabled={disabled}
      aria-label="Change assistant model"
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        minWidth: 0,
        maxWidth: 180,
        border: 'none',
        background: 'transparent',
        padding: 0,
        cursor: disabled ? 'default' : 'pointer',
      }}
    >
      <Typography.Text
        size="sm"
        color="secondary"
        css={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
      >
        {selectedModel}
      </Typography.Text>
      {!disabled && <ChevronDownIcon css={{ fontSize: 12, color: theme.colors.textSecondary }} />}
    </button>
  );

  if (disabled || options.length === 1) {
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
              kind: 'gateway',
              endpointName: provider.model ?? gatewayVendorEndpointName(provider.model_provider),
              gatewayVendor: provider.model_provider,
              providerModel: model,
              modelOptions,
              requiresApiKey: provider.requires_api_key,
              hasApiKey: provider.has_api_key,
            });
          } else {
            onSelect({ kind: 'provider', name: provider.name, model });
          }
        }}
      />
    </>
  );
};
