import { useState } from 'react';
import {
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  PlusIcon,
  Popover,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { useModelsQuery } from '../../../../../gateway/hooks/useModelsQuery';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { CreateEndpointModal } from '../../../../../gateway/components/endpoint-form';
import { sortModelsByDate } from '../../../../../gateway/utils/formatters';
import type { Endpoint } from '../../../../../gateway/types';
import OpenAiLogo from '../../../../../common/static/logos/openai.svg';
import AnthropicLogo from '../../../../../common/static/logos/anthropic.svg';
import GeminiLogo from '../../../../../common/static/logos/gemini.png';
import MLflowGatewayLogo from '../../../../../common/static/logos/mlflow-gateway.svg';

export interface IssueDetectionModelSelection {
  mode: 'endpoint' | 'direct';
  endpointName?: string;
  provider: string;
  model: string;
}

export interface ProviderOption {
  id: string;
  name: string;
  logo: string;
  defaultModel: string;
}

export const ISSUE_DETECTION_PROVIDERS: ProviderOption[] = [
  { id: 'openai', name: 'OpenAI', logo: OpenAiLogo, defaultModel: 'gpt-5.5' },
  { id: 'anthropic', name: 'Anthropic', logo: AnthropicLogo, defaultModel: 'claude-sonnet-4-6' },
  { id: 'gemini', name: 'Google Gemini', logo: GeminiLogo, defaultModel: 'gemini-2.5-pro' },
];

export const GATEWAY_LOGO = MLflowGatewayLogo;

export const ProviderLogo = ({ src }: { src: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <img
      src={src}
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
  const { data: models, isLoading } = useModelsQuery({ provider: isExpanded ? provider.id : undefined });

  // Order models the same way the AI Gateway UI does; fall back to the recommended
  // model if the gateway returns nothing.
  const modelNames = models?.length ? sortModelsByDate(models).map((m) => m.model) : [provider.defaultModel];

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
        <ProviderLogo src={provider.logo} />
        <Typography.Text css={{ flex: 1 }}>{provider.name}</Typography.Text>
      </button>
      {isExpanded &&
        (isLoading ? (
          <div css={{ padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`, paddingLeft: theme.spacing.lg }}>
            <Spinner size="small" />
          </div>
        ) : (
          modelNames.map((model) => (
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
          ))
        ))}
    </div>
  );
};

/**
 * A single collapsible dropdown for choosing the model powering issue detection.
 * Rows are complete choices: an AI Gateway endpoint, or a core provider expanded
 * to its models. Never asks for API keys.
 */
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
  const [open, setOpen] = useState(false);
  // Which group is expanded: a provider id, the literal 'gateway', or null (all collapsed)
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  // Reset to all-collapsed each time the dropdown opens
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
  const triggerLabel = isEndpoint ? value.endpointName : (selectedProvider?.name ?? value.provider);

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
          {triggerLogo && <ProviderLogo src={triggerLogo} />}
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
        // Keep the dropdown anchored below the card. Without this it flips above
        // when a provider group is expanded and the content grows.
        avoidCollisions={false}
        style={{ padding: 0, minWidth: 260 }}
      >
        <div
          css={{
            // Cap to the space available below the trigger (minus a margin from the
            // window edge) so expanding a group scrolls internally instead of growing
            // the popover past the viewport.
            maxHeight: 'min(320px, calc(var(--radix-popover-content-available-height, 320px) - 24px))',
            overflowY: 'auto',
            paddingTop: theme.spacing.xs,
            paddingBottom: theme.spacing.xs,
          }}
        >
          {ISSUE_DETECTION_PROVIDERS.map((provider) => (
            <ProviderGroup
              key={provider.id}
              provider={provider}
              isExpanded={expandedGroup === provider.id}
              selectedModel={value.mode === 'direct' && value.provider === provider.id ? value.model : undefined}
              onToggle={() => toggleGroup(provider.id)}
              onSelectModel={(model) => {
                onChange({ mode: 'direct', provider: provider.id, model });
                setOpen(false);
              }}
            />
          ))}
          <GatewayGroup
            endpoints={endpoints}
            isExpanded={expandedGroup === 'gateway'}
            selectedEndpointName={isEndpoint ? value.endpointName : undefined}
            onToggle={() => toggleGroup('gateway')}
            onSelectEndpoint={selectEndpoint}
            onCreateEndpoint={() => setIsCreateModalOpen(true)}
          />
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
