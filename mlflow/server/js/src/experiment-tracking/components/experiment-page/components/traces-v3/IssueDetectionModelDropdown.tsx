import { useMemo, useState } from 'react';
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  KeyIcon,
  PlusIcon,
  Popover,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import {
  useAllowlistedModelPairs,
  type AllowlistedModelPair,
} from '../../../../../gateway/hooks/useAllowlistedModelPairs';
import { CreateEndpointModal } from '../../../../../gateway/components/endpoint-form';
import type { Endpoint } from '../../../../../gateway/types';
import { getRequiredDefaultLLMProvider, type MlflowDefaultLLMProvider } from '../../../../../gateway/defaultModels';
import OpenAiLogo from '../../../../../common/static/logos/openai.svg';
import OpenAiLogoDark from '../../../../../common/static/logos/openai-dark.svg';
import AnthropicLogo from '../../../../../common/static/logos/anthropic.svg';
import AnthropicLogoDark from '../../../../../common/static/logos/anthropic-dark.png';
import GeminiLogo from '../../../../../common/static/logos/gemini.png';
import MLflowGatewayLogo from '../../../../../common/static/logos/mlflow-gateway.svg';

export interface IssueDetectionModelSelection {
  mode: 'endpoint' | 'direct';
  endpointName?: string;
  provider: string;
  model: string;
  /**
   * The connection (secret) backing a `direct` selection, when the user picked a model from a
   * registered LLM connection. Lets the submit path use that connection's key rather than guessing.
   */
  secretId?: string;
}

export interface ProviderOption {
  id: string;
  name: string;
  logo: string;
  logoDark?: string;
  defaultModel: string;
  models: MlflowDefaultLLMProvider['models'];
}

const openAIProvider = getRequiredDefaultLLMProvider('openai');
const anthropicProvider = getRequiredDefaultLLMProvider('anthropic');
const geminiProvider = getRequiredDefaultLLMProvider('gemini');

export const ISSUE_DETECTION_PROVIDERS: ProviderOption[] = [
  {
    id: openAIProvider.provider,
    name: 'OpenAI',
    logo: OpenAiLogo,
    logoDark: OpenAiLogoDark,
    defaultModel: openAIProvider.defaultModel,
    models: openAIProvider.models,
  },
  {
    id: anthropicProvider.provider,
    name: 'Anthropic',
    logo: AnthropicLogo,
    logoDark: AnthropicLogoDark,
    defaultModel: anthropicProvider.defaultModel,
    models: anthropicProvider.models,
  },
  {
    id: geminiProvider.provider,
    name: 'Google Gemini',
    logo: GeminiLogo,
    defaultModel: geminiProvider.defaultModel,
    models: geminiProvider.models,
  },
];

export const GATEWAY_LOGO = MLflowGatewayLogo;

/**
 * Providers whose connection Detect Issues can run in direct mode (no AI Gateway endpoint) on ANY
 * MLflow install. Two backend steps must both support the provider: credential injection
 * (`_CORE_PROVIDER_ENV_VARS` in mlflow/utils/providers.py) AND the actual call. When LiteLLM is not
 * installed, the call falls back to MLflow's own provider adapters (`_get_provider_instance` in
 * mlflow/metrics/genai/model_utils.py), which cover only this set. Providers that need LiteLLM to
 * call (e.g. groq, deepseek, xai, openrouter) are excluded so a connection can't be picked into a
 * run that would fail on a LiteLLM-less server. Anything else must go through an AI Gateway endpoint.
 */
export const ISSUE_DETECTION_DIRECT_PROVIDERS = new Set([
  'openai',
  'azure',
  'anthropic',
  'bedrock',
  'gemini',
  'mistral',
  'togetherai',
]);

export const getIssueDetectionDirectPairs = (pairs: AllowlistedModelPair[]) =>
  pairs.filter((pair) => ISSUE_DETECTION_DIRECT_PROVIDERS.has(pair.provider.toLowerCase()));

export const ProviderLogo = ({ src, srcDark }: { src: string; srcDark?: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <img
      src={theme.isDarkMode && srcDark ? srcDark : src}
      alt=""
      css={{ width: 18, height: 18, objectFit: 'contain', borderRadius: theme.borders.borderRadiusSm }}
    />
  );
};

const optionRowCss = (theme: ReturnType<typeof useDesignSystemTheme>['theme']) =>
  ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
    width: '100%',
    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    textAlign: 'left' as const,
    '&:hover': { backgroundColor: theme.colors.actionTertiaryBackgroundHover },
  }) as const;

const GatewayGroup = ({
  endpoints,
  isExpanded,
  selectedEndpointName,
  onToggle,
  onSelectEndpoint,
  onCreateEndpoint,
}: {
  endpoints: Endpoint[];
  isExpanded: boolean;
  selectedEndpointName?: string;
  onToggle: () => void;
  onSelectEndpoint: (endpointName: string) => void;
  onCreateEndpoint: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div>
      <button
        type="button"
        onClick={onToggle}
        css={optionRowCss(theme)}
        aria-expanded={isExpanded}
        data-testid="model-group-gateway"
      >
        {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <ProviderLogo src={GATEWAY_LOGO} />
        <Typography.Text css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="AI Gateway" description="Model dropdown group for AI Gateway endpoints" />
        </Typography.Text>
      </button>
      {isExpanded && (
        <>
          {endpoints.map((endpoint) => (
            <button
              key={endpoint.name}
              type="button"
              onClick={() => onSelectEndpoint(endpoint.name)}
              css={{ ...optionRowCss(theme), paddingLeft: 44 }}
              data-testid={`model-option-endpoint-${endpoint.name}`}
            >
              <Typography.Text css={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {endpoint.name}
              </Typography.Text>
              {selectedEndpointName === endpoint.name && (
                <CheckIcon css={{ color: theme.colors.actionDefaultBorderFocus }} />
              )}
            </button>
          ))}
          <button
            type="button"
            onClick={onCreateEndpoint}
            css={{ ...optionRowCss(theme), paddingLeft: 44, color: theme.colors.actionTertiaryTextDefault }}
            data-testid="model-create-endpoint"
          >
            <PlusIcon />
            <Typography.Text color="info">
              <FormattedMessage
                defaultMessage="Create endpoint"
                description="Action in the model dropdown to create a new AI Gateway endpoint"
              />
            </Typography.Text>
          </button>
        </>
      )}
    </div>
  );
};

const ConnectionsGroup = ({
  pairs,
  isExpanded,
  selectedSecretId,
  selectedModel,
  onToggle,
  onSelectPair,
}: {
  pairs: AllowlistedModelPair[];
  isExpanded: boolean;
  selectedSecretId?: string;
  selectedModel?: string;
  onToggle: () => void;
  onSelectPair: (pair: AllowlistedModelPair) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div>
      <button
        type="button"
        onClick={onToggle}
        css={optionRowCss(theme)}
        aria-expanded={isExpanded}
        data-testid="model-group-connections"
      >
        {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <KeyIcon css={{ color: theme.colors.textSecondary }} />
        <Typography.Text css={{ flex: 1 }}>
          <FormattedMessage
            defaultMessage="Existing connections"
            description="Model dropdown group for models from registered LLM connections"
          />
        </Typography.Text>
      </button>
      {isExpanded &&
        pairs.map((pair) => (
          <button
            key={`${pair.secretId}::${pair.model}`}
            type="button"
            onClick={() => onSelectPair(pair)}
            css={{ ...optionRowCss(theme), paddingLeft: 44 }}
            data-testid={`model-option-connection-${pair.secretId}-${pair.model}`}
          >
            <Typography.Text css={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {pair.model} · {pair.secretName}
            </Typography.Text>
            {selectedSecretId === pair.secretId && selectedModel === pair.model && (
              <CheckIcon css={{ color: theme.colors.actionDefaultBorderFocus }} />
            )}
          </button>
        ))}
    </div>
  );
};

const ProviderGroup = ({
  provider,
  isExpanded,
  selectedModel,
  onToggle,
  onSelectModel,
}: {
  provider: ProviderOption;
  isExpanded: boolean;
  selectedModel?: string;
  onToggle: () => void;
  onSelectModel: (model: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div>
      <button
        type="button"
        onClick={onToggle}
        css={optionRowCss(theme)}
        aria-expanded={isExpanded}
        data-testid={`model-provider-${provider.id}`}
      >
        {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <ProviderLogo src={provider.logo} srcDark={provider.logoDark} />
        <Typography.Text css={{ flex: 1 }}>{provider.name}</Typography.Text>
      </button>
      {isExpanded &&
        provider.models.map(({ model }) => (
          <button
            key={model}
            type="button"
            onClick={() => onSelectModel(model)}
            css={{ ...optionRowCss(theme), paddingLeft: 44 }}
            data-testid={`model-option-${provider.id}-${model}`}
          >
            <Typography.Text css={{ flex: 1 }}>{model}</Typography.Text>
            {selectedModel === model && <CheckIcon css={{ color: theme.colors.actionDefaultBorderFocus }} />}
          </button>
        ))}
    </div>
  );
};

export const IssueDetectionModelDropdown = ({
  endpoints,
  value,
  onChange,
}: {
  endpoints: Endpoint[];
  value: IssueDetectionModelSelection;
  onChange: (value: IssueDetectionModelSelection) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { refetch: refetchEndpoints } = useEndpointsQuery();
  const { pairs } = useAllowlistedModelPairs();
  const [open, setOpen] = useState(false);
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  // Only connections whose provider Detect Issues can resolve in direct mode are selectable here.
  // A non-core provider can only be reached through an AI Gateway endpoint (see the constant above).
  const directPairs = useMemo(() => getIssueDetectionDirectPairs(pairs), [pairs]);

  // The currently-selected registered connection (if any), used to label the trigger "name · model".
  const selectedConnectionName =
    value.mode === 'direct' && value.secretId
      ? directPairs.find((p) => p.secretId === value.secretId && p.model === value.model)?.secretName
      : undefined;

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (nextOpen) {
      setExpandedGroup(null);
    }
  };

  const toggleGroup = (id: string) => setExpandedGroup((current) => (current === id ? null : id));

  const selectEndpoint = (endpointName: string) => {
    onChange({
      mode: 'endpoint',
      endpointName,
      provider: ISSUE_DETECTION_PROVIDERS[0].id,
      model: ISSUE_DETECTION_PROVIDERS[0].defaultModel,
    });
    setOpen(false);
  };

  const isEndpoint = value.mode === 'endpoint';
  const selectedProvider = ISSUE_DETECTION_PROVIDERS.find((p) => p.id === value.provider);
  const triggerLogo = isEndpoint ? GATEWAY_LOGO : selectedProvider?.logo;
  const triggerLogoDark = isEndpoint ? undefined : selectedProvider?.logoDark;
  const triggerLabel = isEndpoint
    ? value.endpointName
    : (selectedConnectionName ?? selectedProvider?.name ?? value.provider);

  return (
    <Popover.Root
      componentId="mlflow.traces.issue-detection-modal.model-dropdown"
      open={open}
      onOpenChange={handleOpenChange}
    >
      <Popover.Trigger asChild>
        <button
          type="button"
          data-testid="model-dropdown-trigger"
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            width: '100%',
            padding: theme.spacing.sm,
            background: 'none',
            textAlign: 'left',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            cursor: 'pointer',
            '&:hover': {
              backgroundColor: theme.colors.actionTertiaryBackgroundHover,
              borderColor: theme.colors.actionDefaultBorderHover,
            },
          }}
        >
          {triggerLogo && <ProviderLogo src={triggerLogo} srcDark={triggerLogoDark} />}
          <div css={{ minWidth: 0, flex: 1 }}>
            <Typography.Text css={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {triggerLabel}
            </Typography.Text>
            {!isEndpoint && value.model && <Typography.Hint>{value.model}</Typography.Hint>}
          </div>
          <ChevronDownIcon css={{ color: theme.colors.textSecondary }} />
        </button>
      </Popover.Trigger>
      <Popover.Content
        side="bottom"
        align="start"
        sideOffset={4}
        collisionPadding={16}
        avoidCollisions={false}
        style={{ padding: 0, minWidth: 260 }}
      >
        <div
          css={{
            maxHeight: 'min(320px, calc(var(--radix-popover-content-available-height, 320px) - 24px))',
            overflowY: 'auto',
            paddingTop: theme.spacing.xs,
            paddingBottom: theme.spacing.xs,
          }}
        >
          {directPairs.length > 0 && (
            <ConnectionsGroup
              pairs={directPairs}
              isExpanded={expandedGroup === 'connections'}
              selectedSecretId={value.mode === 'direct' ? value.secretId : undefined}
              selectedModel={value.mode === 'direct' ? value.model : undefined}
              onToggle={() => toggleGroup('connections')}
              onSelectPair={(pair) => {
                onChange({ mode: 'direct', provider: pair.provider, model: pair.model, secretId: pair.secretId });
                setOpen(false);
              }}
            />
          )}
          <GatewayGroup
            endpoints={endpoints}
            isExpanded={expandedGroup === 'gateway'}
            selectedEndpointName={isEndpoint ? value.endpointName : undefined}
            onToggle={() => toggleGroup('gateway')}
            onSelectEndpoint={selectEndpoint}
            onCreateEndpoint={() => setIsCreateModalOpen(true)}
          />
          {ISSUE_DETECTION_PROVIDERS.map((provider) => (
            <ProviderGroup
              key={provider.id}
              provider={provider}
              isExpanded={expandedGroup === provider.id}
              selectedModel={
                value.mode === 'direct' && !value.secretId && value.provider === provider.id ? value.model : undefined
              }
              onToggle={() => toggleGroup(provider.id)}
              onSelectModel={(model) => {
                onChange({ mode: 'direct', provider: provider.id, model });
                setOpen(false);
              }}
            />
          ))}
        </div>
      </Popover.Content>
      {isCreateModalOpen && (
        <CreateEndpointModal
          open={isCreateModalOpen}
          onClose={() => setIsCreateModalOpen(false)}
          onSuccess={(endpoint) => {
            refetchEndpoints();
            selectEndpoint(endpoint.name);
            setIsCreateModalOpen(false);
          }}
        />
      )}
    </Popover.Root>
  );
};
