/**
 * Assistant Chat Panel component.
 * Displays the chat interface for the Assistant.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Alert,
  Button,
  Card,
  CheckIcon,
  ChevronDownIcon,
  CloseIcon,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  DropdownMenu,
  GearIcon,
  InfoTooltip,
  PlusIcon,
  RefreshIcon,
  SparkleDoubleIcon,
  SparkleIcon,
  StopIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  SendIcon,
  WrenchSparkleIcon,
  Spinner,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';
import { useAssistantPageContext } from './AssistantPageContext';
import { getAssistantProvider, getLlmProviderDisplay } from './providerRegistry';
import { AssistantErrorCode, type ProviderInfo, type SelectProviderOptions, type SelectedProvider } from './types';
import { ApiKeyPrompt } from './ApiKeyPrompt';
import { GATEWAY_PROVIDER_ID } from './constants';
import { useEndpointsQuery } from '../gateway/hooks/useEndpointsQuery';
import { GatewayRoutePaths } from '../gateway/routes';
import type { Endpoint } from '../gateway/types';
import { AssistantContextTags } from './AssistantContextTags';
import { ToolPermissionPrompt } from './ToolPermissionPrompt';
import { ToolCallGroup, type ToolCallPart } from './ToolCallCard';
import type { AssistantPart, ChatMessage } from './types';
import { AssistantSetupWizard } from './setup';
import { useLogTelemetryEvent } from '../telemetry/hooks/useLogTelemetryEvent';
import { GenAIMarkdownRenderer } from '../shared/web-shared/genai-markdown-renderer';
import { useCopyController } from '../shared/web-shared/snippet/hooks/useCopyController';
import { useAssistantPrompts } from '../common/utils/RoutingUtils';
import { AssistantWelcomeCarousel } from './AssistantWelcomeCarousel';

type CurrentView = 'chat' | 'setup-wizard' | 'settings';

// Shared animation keyframes
const PULSE_ANIMATION = {
  '0%, 100%': { transform: 'scale(1)' },
  '50%': { transform: 'scale(1.3)' },
};

const DOTS_ANIMATION = {
  '0%': { content: '""' },
  '33%': { content: '"."' },
  '66%': { content: '".."' },
  '100%': { content: '"..."' },
};

// Abbreviate token counts for the compact usage footer (e.g. 45257 -> "45.3K").
const formatCompactTokens = (n: number): string =>
  new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(n);

// Sub-dollar estimates need more precision than cents (e.g. "$0.0045").
const formatCostUsd = (cost: number): string =>
  new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: cost < 1 ? 4 : 2,
  }).format(cost);

const GATEWAY_VENDOR_ORDER = ['openai', 'anthropic', 'gemini'];

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

/**
 * Endpoint list inside the gateway submenu, plus a shortcut to create a new
 * endpoint. Lives in its own component so the endpoints fetch only fires when
 * the submenu is actually opened.
 */
const GatewayEndpointItems = ({
  currentProvider,
  gatewayVendorOptions,
  onSelect,
}: {
  currentProvider: SelectedProvider;
  gatewayVendorOptions: Record<string, string[]>;
  onSelect: (endpointName: string, options?: SelectProviderOptions) => void;
}) => {
  const { data: endpoints, isLoading } = useEndpointsQuery();
  const { theme } = useDesignSystemTheme();
  const managedEndpointNames = new Set(Object.keys(gatewayVendorOptions).map(gatewayVendorEndpointName));
  const managedVendors = orderedGatewayVendors(gatewayVendorOptions);
  const customEndpoints = endpoints.filter((endpoint) => !managedEndpointNames.has(endpoint.name));

  const openCreateEndpoint = useCallback(() => {
    // MLflow uses hash routing, so SPA routes must be prefixed with `/#`
    // for fresh-tab loads to land on the right page.
    window.open(`/#${GatewayRoutePaths.createEndpointPage}`, '_blank', 'noopener');
  }, []);

  return (
    <>
      {managedVendors.map((vendor) => {
        const endpointName = gatewayVendorEndpointName(vendor);
        const endpoint = endpoints.find((candidate) => candidate.name === endpointName);
        const modelOptions = gatewayVendorOptions[vendor] ?? [];
        const providerModel = endpointModelName(endpoint) ?? modelOptions[0];
        const vendorDisplay = getLlmProviderDisplay(vendor);
        const isCurrent =
          currentProvider.id === GATEWAY_PROVIDER_ID &&
          currentProvider.model === endpointName &&
          currentProvider.modelProvider === vendor;

        return (
          <DropdownMenu.Item
            componentId="mlflow.assistant.provider_picker.item"
            key={vendor}
            onClick={() =>
              onSelect(endpointName, {
                gatewayVendor: vendor,
                modelProvider: vendor,
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
            {isCurrent && <CheckIcon css={{ marginLeft: theme.spacing.sm }} />}
          </DropdownMenu.Item>
        );
      })}
      {managedVendors.length > 0 && <DropdownMenu.Separator />}
      {isLoading ? (
        <DropdownMenu.Item componentId="mlflow.assistant.provider_picker.endpoint" disabled>
          <Spinner size="small" />
        </DropdownMenu.Item>
      ) : (
        customEndpoints.map((endpoint) => (
          <DropdownMenu.Item
            componentId="mlflow.assistant.provider_picker.endpoint"
            key={endpoint.name}
            onClick={() => onSelect(endpoint.name)}
          >
            {endpoint.name}
          </DropdownMenu.Item>
        ))
      )}
      <DropdownMenu.Item componentId="mlflow.assistant.provider_picker.create_endpoint" onClick={openCreateEndpoint}>
        <PlusIcon css={{ marginRight: 4 }} />
        <FormattedMessage
          defaultMessage="Create new endpoint"
          description="Option in the assistant provider picker that opens the gateway endpoint creation page"
        />
      </DropdownMenu.Item>
    </>
  );
};

/** Hover explanation for providers that cannot be selected right now. */
const disabledReasonFor = (candidate: ProviderInfo, intl: ReturnType<typeof useIntl>): string => {
  // These CLI providers are localhost-only, so they only ever appear on the same
  // machine that runs the server — hence "your machine", not "the MLflow server".
  if (candidate.name === 'claude_code' || candidate.name === 'codex') {
    return intl.formatMessage(
      {
        defaultMessage: 'The {name} CLI is not installed on your machine.',
        description: 'Hover explanation for a CLI assistant provider that is not installed',
      },
      { name: candidate.display_name },
    );
  }
  if (candidate.name === 'ollama') {
    return intl.formatMessage({
      defaultMessage: 'No running Ollama server was found (or it has no models pulled).',
      description: 'Hover explanation for the Ollama assistant provider when unavailable',
    });
  }
  return intl.formatMessage({
    defaultMessage: 'This provider is not available on the MLflow server.',
    description: 'Generic hover explanation for an unavailable assistant provider',
  });
};

/**
 * Provider chip in the composer: shows the provider/vendor backing the session
 * and opens an upward menu to switch to any other discovered provider without
 * going through Settings. Switching is optimistic and local (no backend call);
 * the selection is persisted on the next send, and anything the new provider
 * still needs (API key, login) is collected lazily then.
 */
const ProviderPicker = ({
  provider,
  providers,
  gatewayVendorOptions,
  disabled,
  onSelect,
}: {
  provider: SelectedProvider;
  providers: ProviderInfo[];
  gatewayVendorOptions: Record<string, string[]>;
  disabled?: boolean;
  onSelect: (providerId: string, model?: string, options?: SelectProviderOptions) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const meta = getAssistantProvider(provider.id);
  // The gateway routes rather than serves models: show the endpoint's actual
  // LLM vendor (OpenAI, Anthropic, ...) when it's known.
  const llmVendor = provider.modelProvider ? getLlmProviderDisplay(provider.modelProvider) : undefined;
  const label = llmVendor?.name ?? meta?.name ?? provider.id;
  const logo = llmVendor?.logo ?? meta?.logo;

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
      <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
        {label}
      </Typography.Text>
      {!disabled && <ChevronDownIcon css={{ fontSize: 12, color: theme.colors.textSecondary }} />}
    </button>
  );

  if (disabled) {
    // Remote clients cannot update the server-side config, so the chip stays display-only.
    return trigger;
  }

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Content side="top" align="start" css={{ minWidth: 220 }}>
        {providers.map((candidate) => {
          const candidateMeta = getAssistantProvider(candidate.name);
          const candidateLogo = candidateMeta?.logo;
          const candidateLabel = candidateMeta?.name ?? candidate.display_name;
          const isCurrent = candidate.name === provider.id;
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
              <span css={{ flex: 1 }}>{candidateLabel}</span>
              {isCurrent && <CheckIcon css={{ marginLeft: theme.spacing.sm }} />}
            </>
          );
          if (candidate.name === GATEWAY_PROVIDER_ID) {
            // Always enabled: the submenu lists existing endpoints and offers
            // creating a new one when there are none yet.
            return (
              <DropdownMenu.Sub key={candidate.name}>
                <DropdownMenu.SubTrigger>{itemContent}</DropdownMenu.SubTrigger>
                <DropdownMenu.SubContent>
                  <GatewayEndpointItems
                    currentProvider={provider}
                    gatewayVendorOptions={gatewayVendorOptions}
                    onSelect={(endpointName, options) => onSelect(candidate.name, endpointName, options)}
                  />
                </DropdownMenu.SubContent>
              </DropdownMenu.Sub>
            );
          }
          return (
            <DropdownMenu.Item
              componentId="mlflow.assistant.provider_picker.item"
              key={candidate.name}
              disabled={!candidate.available}
              disabledReason={candidate.available ? undefined : disabledReasonFor(candidate, intl)}
              onClick={() => onSelect(candidate.name, candidate.model_options[0])}
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

  const currentModel = model && model !== 'default' && options.includes(model) ? model : options[0];
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
        {currentModel}
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
            {option === currentModel && <CheckIcon css={{ marginLeft: theme.spacing.sm }} />}
          </DropdownMenu.Item>
        ))}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

export type MessagePartGroup = { kind: 'text'; text: string } | { kind: 'tools'; calls: ToolCallPart[] };

/**
 * Coalesces an ordered part list into render groups, collapsing each maximal run of
 * adjacent tool calls into a single `tools` group while preserving interleaving order.
 */
export const groupParts = (parts: AssistantPart[]): MessagePartGroup[] => {
  const groups: MessagePartGroup[] = [];
  for (const part of parts) {
    if (part.type === 'text') {
      groups.push({ kind: 'text', text: part.text });
      continue;
    }
    const last = groups[groups.length - 1];
    if (last?.kind === 'tools') {
      last.calls = [...last.calls, part];
    } else {
      groups.push({ kind: 'tools', calls: [part] });
    }
  }
  return groups;
};

/**
 * Renders an assistant turn's ordered parts (text + tool calls). Falls back to plain
 * `content` for messages that predate the parts model.
 */
export const AssistantMessageBody = ({ message }: { message: ChatMessage }) => {
  const { theme } = useDesignSystemTheme();
  const parts: AssistantPart[] = message.parts ?? [{ type: 'text', text: message.content }];
  // The markdown renderer leaves `---` as a default <hr> and headings with tight margins,
  // so model-generated section breaks read cramped. Give rules and headings room to breathe.
  const markdownSpacing = {
    '& hr': {
      margin: `${theme.spacing.lg}px 0`,
      border: 'none',
      borderTop: `1px solid ${theme.colors.border}`,
    },
    '& h1, & h2, & h3, & h4': { marginTop: theme.spacing.md },
  };
  return (
    <>
      {groupParts(parts).map((group, i) =>
        group.kind === 'text' ? (
          group.text ? (
            <div key={`text-${i}`} css={markdownSpacing}>
              <GenAIMarkdownRenderer>{group.text}</GenAIMarkdownRenderer>
            </div>
          ) : null
        ) : (
          <ToolCallGroup key={group.calls[0].toolUseId} parts={group.calls} />
        ),
      )}
    </>
  );
};

/** Animated "working" indicator shown while the assistant streams a turn. */
const StreamingIndicator = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        marginTop: theme.spacing.sm,
      }}
    >
      <SparkleIcon
        color="ai"
        css={{ fontSize: 16, animation: 'pulse 1.5s ease-in-out infinite', '@keyframes pulse': PULSE_ANIMATION }}
      />
      <span
        css={{
          fontSize: theme.typography.fontSizeBase,
          color: theme.colors.textSecondary,
          '&::after': {
            content: '"..."',
            animation: 'dots 1.5s steps(3, end) infinite',
            display: 'inline-block',
            width: '1.2em',
          },
          '@keyframes dots': DOTS_ANIMATION,
        }}
      >
        Processing
      </span>
    </div>
  );
};

/**
 * Single chat message bubble.
 */
const ChatMessageBubble = ({
  message,
  isLastMessage,
  onRegenerate,
}: {
  message: ChatMessage;
  isLastMessage: boolean;
  onRegenerate?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const isUser = message.role === 'user';
  const [isHovered, setIsHovered] = useState(false);
  const { actionIcon: copyIcon, tooltipMessage: copyTooltip, copy: handleCopy } = useCopyController(message.content);

  const handleRegenerate = () => {
    onRegenerate?.();
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: isUser ? theme.spacing.md : 0,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        css={{
          maxWidth: isUser ? '85%' : '100%',
          padding: `${theme.spacing.md}px ${theme.spacing.md}px ${isUser ? theme.spacing.md : 0}px ${
            theme.spacing.md
          }px`,
          borderRadius: theme.borders.borderRadiusLg,
          backgroundColor: isUser ? theme.colors.backgroundSecondary : 'transparent',
          color: theme.colors.textPrimary,
        }}
      >
        {isUser ? (
          <Typography.Text css={{ whiteSpace: 'pre-wrap' }}>{message.content}</Typography.Text>
        ) : (
          <AssistantMessageBody message={message} />
        )}
        {/* Interrupted indicator */}
        {message.isInterrupted && (
          <span
            css={{
              display: 'block',
              marginTop: theme.spacing.sm,
              fontSize: theme.typography.fontSizeSm,
              fontStyle: 'italic',
              color: theme.colors.textSecondary,
            }}
          >
            Interrupted by user
          </span>
        )}
        {/* Loading indicator */}
        {message.isStreaming && <StreamingIndicator />}
      </div>

      {/* Action buttons for assistant messages */}
      {!isUser && !message.isStreaming && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
            paddingLeft: theme.spacing.md,
            opacity: isLastMessage || isHovered ? 1 : 0,
            pointerEvents: isLastMessage || isHovered ? 'auto' : 'none',
            transition: 'opacity 0.2s ease',
          }}
        >
          <Tooltip componentId="mlflow.assistant.chat_panel.copy.tooltip" content={copyTooltip}>
            <Button componentId="mlflow.assistant.chat_panel.copy" size="small" icon={copyIcon} onClick={handleCopy} />
          </Tooltip>
          {isLastMessage && (
            <Tooltip componentId="mlflow.assistant.chat_panel.regenerate.tooltip" content="Regenerate">
              <Button
                componentId="mlflow.assistant.chat_panel.regenerate"
                size="small"
                icon={<RefreshIcon />}
                onClick={handleRegenerate}
              />
            </Tooltip>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Initial prompt suggestions empty state.
 */
const PromptSuggestions = ({ onSelect }: { onSelect: (prompt: string) => void }) => {
  const { theme } = useDesignSystemTheme();
  const suggestions = useAssistantPrompts();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing.lg * 2,
        gap: theme.spacing.lg,
      }}
    >
      <WrenchSparkleIcon color="ai" css={{ fontSize: 64, opacity: 0.75 }} />

      <Typography.Text
        color="secondary"
        css={{
          fontSize: theme.typography.fontSizeMd,
          textAlign: 'center',
          maxWidth: 400,
        }}
      >
        Ask questions about your experiments, traces, evaluations, and more.
      </Typography.Text>

      {/* Suggestion cards */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: theme.spacing.sm,
          width: '100%',
          maxWidth: 400,
        }}
      >
        {suggestions.map((suggestion) => (
          <Card
            componentId="mlflow.assistant.chat_panel.suggestion.card"
            key={suggestion}
            onClick={() => onSelect(suggestion)}
            css={{
              cursor: 'pointer',
              padding: theme.spacing.sm,
              transition: 'all 0.2s ease',
              '&:hover': {
                borderColor: theme.colors.actionPrimaryBackgroundDefault,
              },
            }}
          >
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
              }}
            >
              <SparkleIcon color="ai" css={{ fontSize: 18, flexShrink: 0 }} />
              <span>{suggestion}</span>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

/**
 * Chat panel content component.
 */
/**
 * A stream error the user can act on directly (install/login/configure), rendered as a
 * callout above the composer with a shortcut to the settings flow.
 */
const RecoverableErrorCallout = ({
  errorCode,
  error,
  onOpenSettings,
}: {
  errorCode: string;
  error: string | null;
  onOpenSettings: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  let message: React.ReactNode;
  switch (errorCode) {
    case AssistantErrorCode.CliNotInstalled:
      message = (
        <FormattedMessage
          defaultMessage="The provider's CLI is not installed on the MLflow server. Install it, or switch to another provider."
          description="Callout shown when the assistant provider CLI is missing"
        />
      );
      break;
    case AssistantErrorCode.NotAuthenticated:
      message = (
        <FormattedMessage
          defaultMessage="The provider is not authenticated. Fix the credentials, or switch to another provider."
          description="Callout shown when the assistant provider rejected the credentials"
        />
      );
      break;
    default:
      message = (
        <FormattedMessage
          defaultMessage="No assistant provider is available. Configure one in Settings."
          description="Callout shown when no assistant provider could serve the chat"
        />
      );
  }

  return (
    <Alert
      componentId="mlflow.assistant.chat_panel.recoverable_error"
      type="warning"
      closable={false}
      css={{ marginBottom: theme.spacing.sm }}
      message={message}
      description={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, alignItems: 'flex-start' }}>
          {error && (
            <Typography.Text size="sm" color="secondary" css={{ wordBreak: 'break-word' }}>
              {error}
            </Typography.Text>
          )}
          <Button
            componentId="mlflow.assistant.chat_panel.recoverable_error.action"
            size="small"
            onClick={onOpenSettings}
          >
            <FormattedMessage
              defaultMessage="Open Settings"
              description="Button on the assistant error callout that opens the settings flow"
            />
          </Button>
        </div>
      }
    />
  );
};

// api_key_missing is deliberately absent: it renders the inline ApiKeyPrompt instead.
const RECOVERABLE_ERROR_CODES = new Set<string>([
  AssistantErrorCode.CliNotInstalled,
  AssistantErrorCode.NotAuthenticated,
  AssistantErrorCode.NoProvider,
]);

const ChatPanelContent = ({ onOpenSettings }: { onOpenSettings: () => void }) => {
  const { theme } = useDesignSystemTheme();
  const {
    messages,
    isStreaming,
    error,
    errorCode,
    sendMessage,
    regenerateLastMessage,
    cancelSession,
    tokenUsage,
    selectedProvider,
    availableProviders,
    gatewayVendorOptions,
    selectProvider,
    isLocalServer,
    needsApiKey,
    refreshConfig,
    pendingPrompt,
    clearPendingPrompt,
    pendingPermission,
    respondToPermission,
  } = useAssistant();
  const logTelemetryEvent = useLogTelemetryEvent();
  const viewId = useMemo(() => uuidv4(), []);

  const [inputValue, setInputValue] = useState('');
  // A message stashed while the API-key modal collects the missing key; sent on save.
  const [pendingKeyMessage, setPendingKeyMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const currentProviderInfo = selectedProvider
    ? availableProviders.find((provider) => provider.name === selectedProvider.id)
    : undefined;
  const currentModelOptions =
    selectedProvider?.modelOptions && selectedProvider.modelOptions.length > 0
      ? selectedProvider.modelOptions
      : (currentProviderInfo?.model_options ?? []);
  const currentModel =
    selectedProvider?.modelOptions && selectedProvider.modelOptions.length > 0
      ? (selectedProvider.providerModel ?? currentModelOptions[0] ?? selectedProvider.model)
      : (selectedProvider?.model ?? 'default');

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea to fit content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [inputValue]);

  // Consume a prompt queued by an onboarding card. This component only mounts once setup is
  // complete, so a prompt queued during the setup wizard lands here on first mount.
  useEffect(() => {
    if (pendingPrompt != null) {
      setInputValue(pendingPrompt);
      clearPendingPrompt();
      textareaRef.current?.focus();
    }
  }, [pendingPrompt, clearPendingPrompt]);

  // Deliver a message, first collecting the resolved provider's API key when it
  // is still missing (the ideal-flow popup: the first send doubles as setup).
  const deliverMessage = useCallback(
    (message: string) => {
      if (needsApiKey && selectedProvider) {
        setPendingKeyMessage(message);
        return;
      }
      sendMessage(message);
    },
    [needsApiKey, selectedProvider, sendMessage],
  );

  const handleApiKeySaved = useCallback(() => {
    const message = pendingKeyMessage;
    setPendingKeyMessage(null);
    // Send right away: this clears any api_key_missing error in the same render,
    // so the prompt hides immediately instead of lingering while a refresh runs.
    // (The key is already saved server-side; the send carries it.) If the prompt
    // came from a failed send rather than a queued message, retry that turn.
    if (message) {
      sendMessage(message);
    } else {
      regenerateLastMessage();
    }
    // Update the resolved-provider / needsApiKey state in the background.
    refreshConfig();
  }, [pendingKeyMessage, sendMessage, regenerateLastMessage, refreshConfig]);

  // If the key prompt is dismissed sideways (e.g. the user switches provider from
  // the picker instead of entering a key), put the stashed message back in the
  // input so nothing typed is lost.
  useEffect(() => {
    if (pendingKeyMessage != null && !needsApiKey) {
      setInputValue((current) => current || pendingKeyMessage);
      setPendingKeyMessage(null);
    }
  }, [needsApiKey, pendingKeyMessage]);

  const handleSend = useCallback(() => {
    if (inputValue.trim() && !isStreaming) {
      deliverMessage(inputValue.trim());
      setInputValue('');
    }
  }, [inputValue, isStreaming, deliverMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (inputValue.trim() && !isStreaming) {
          logTelemetryEvent({
            componentId: 'mlflow.assistant.chat_panel.send',
            componentViewId: viewId,
            componentType: DesignSystemEventProviderComponentTypes.Button,
            componentSubType: null,
            eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
          });
        }
        handleSend();
      }
    },
    [handleSend, inputValue, isStreaming, logTelemetryEvent, viewId],
  );

  const handleSuggestionSelect = useCallback(
    (prompt: string) => {
      deliverMessage(prompt);
    },
    [deliverMessage],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
      }}
    >
      {/* Messages area */}
      <div
        css={{
          flex: 1,
          overflowY: 'auto',
          padding: theme.spacing.lg,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: messages.length === 0 && !isStreaming ? 'center' : 'flex-start',
          alignItems: messages.length === 0 && !isStreaming ? 'center' : 'stretch',
        }}
        onWheel={(e) => e.stopPropagation()}
      >
        {messages.length === 0 && !isStreaming && <PromptSuggestions onSelect={handleSuggestionSelect} />}

        {messages.map((message, index) => {
          // Check if this is the last assistant message
          const isLastAssistantMessage = message.role === 'assistant' && index === messages.length - 1;
          return (
            <ChatMessageBubble
              key={message.id}
              message={message}
              isLastMessage={isLastAssistantMessage}
              onRegenerate={isLastAssistantMessage ? regenerateLastMessage : undefined}
            />
          );
        })}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px ${theme.spacing.md}px ${theme.spacing.md}px`,
          flexShrink: 0,
        }}
      >
        {pendingPermission && <ToolPermissionPrompt request={pendingPermission} onRespond={respondToPermission} />}
        {errorCode && RECOVERABLE_ERROR_CODES.has(errorCode) && (
          <RecoverableErrorCallout errorCode={errorCode} error={error} onOpenSettings={onOpenSettings} />
        )}
        {(pendingKeyMessage != null || errorCode === AssistantErrorCode.ApiKeyMissing) && selectedProvider && (
          <ApiKeyPrompt provider={selectedProvider} onSaved={handleApiKeySaved} />
        )}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`,
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          <AssistantContextTags />
          <textarea
            ref={textareaRef}
            placeholder="Ask a question..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            css={{
              width: '100%',
              border: 'none',
              outline: 'none',
              backgroundColor: 'transparent',
              fontSize: theme.typography.fontSizeBase,
              color: theme.colors.textPrimary,
              padding: theme.spacing.xs,
              resize: 'none',
              overflowX: 'hidden',
              overflowY: 'auto',
              fontFamily: 'inherit',
              lineHeight: 'inherit',
              maxHeight: 150,
              '&::placeholder': {
                color: theme.colors.textPlaceholder,
              },
              '&:focus': {
                border: 'none',
                outline: 'none',
              },
            }}
          />
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              paddingTop: theme.spacing.sm,
              marginTop: theme.spacing.xs,
              borderTop: `1px solid ${theme.colors.borderDecorative}`,
            }}
          >
            {selectedProvider && (
              <>
                <ProviderPicker
                  provider={selectedProvider}
                  providers={availableProviders}
                  gatewayVendorOptions={gatewayVendorOptions}
                  disabled={!isLocalServer}
                  onSelect={selectProvider}
                />
                <ModelPicker
                  model={currentModel}
                  options={currentModelOptions}
                  disabled={!isLocalServer}
                  onSelect={(model) => {
                    if (selectedProvider.id === GATEWAY_PROVIDER_ID && selectedProvider.modelProvider) {
                      selectProvider(selectedProvider.id, selectedProvider.model, {
                        gatewayVendor: selectedProvider.modelProvider,
                        modelProvider: selectedProvider.modelProvider,
                        providerModel: model,
                        modelOptions: currentModelOptions,
                        requiresApiKey: selectedProvider.requiresApiKey,
                        hasApiKey: selectedProvider.hasApiKey,
                      });
                    } else {
                      selectProvider(selectedProvider.id, model);
                    }
                  }}
                />
              </>
            )}
            <div css={{ flex: 1 }} />
            {tokenUsage.totalTokens > 0 && (
              <div css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Typography.Text size="sm" color="secondary" css={{ fontVariantNumeric: 'tabular-nums' }}>
                  {formatCompactTokens(tokenUsage.totalTokens)}
                </Typography.Text>
                <InfoTooltip
                  componentId="mlflow.assistant.chat_panel.usage_info"
                  content={
                    <div
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.xs,
                        fontVariantNumeric: 'tabular-nums',
                      }}
                    >
                      <span>
                        <FormattedMessage
                          defaultMessage="Input {input} · Output {output} tokens"
                          description="Breakdown of session token usage into input (prompt) and output (completion) tokens"
                          values={{
                            input: tokenUsage.promptTokens.toLocaleString(),
                            output: tokenUsage.completionTokens.toLocaleString(),
                          }}
                        />
                      </span>
                      {tokenUsage.costUsd != null ? (
                        <span>
                          <FormattedMessage
                            defaultMessage="Estimated cost ~{cost} — from public model pricing; actual may vary (provider and cache rates)."
                            description="Estimated session cost with a disclaimer that it is approximate"
                            values={{ cost: formatCostUsd(tokenUsage.costUsd) }}
                          />
                        </span>
                      ) : (
                        <span>
                          <FormattedMessage
                            defaultMessage="Cost estimate unavailable for this model."
                            description="Shown when the assistant's model is not in the pricing catalog so cost cannot be estimated"
                          />
                        </span>
                      )}
                    </div>
                  }
                />
              </div>
            )}
            <Button
              componentId="mlflow.assistant.chat_panel.send"
              onClick={isStreaming ? cancelSession : handleSend}
              disabled={!isStreaming && !inputValue.trim()}
              icon={isStreaming ? <StopIcon /> : <SendIcon />}
              aria-label="Send message"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Loading state while fetching setup status.
 */
const SetupLoadingState = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        gap: theme.spacing.md,
      }}
    >
      <Spinner size="default" />
      <Typography.Text color="secondary">Loading...</Typography.Text>
    </div>
  );
};

/**
 * Message shown when this client is not allowed to use the Assistant,
 * e.g. a remote client when the server's remote-access settings don't permit it.
 */
const RemoteServerMessage = ({ onClose }: { onClose: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
        minHeight: 0,
        padding: theme.spacing.lg,
        paddingBottom: theme.spacing.lg * 3,
        gap: theme.spacing.lg,
      }}
    >
      <WrenchSparkleIcon color="ai" css={{ fontSize: 64, opacity: 0.5 }} />

      <Typography.Title level={4} css={{ textAlign: 'center', marginBottom: 0 }}>
        <FormattedMessage
          defaultMessage="Assistant Not Available"
          description="Title shown when Assistant is not available for this client"
        />
      </Typography.Title>

      <Typography.Text
        color="secondary"
        css={{
          fontSize: theme.typography.fontSizeMd,
          textAlign: 'center',
          maxWidth: 400,
        }}
      >
        <FormattedMessage
          defaultMessage="MLflow Assistant is not available from this client. Ask your MLflow server administrator to enable remote access if you need to use it remotely."
          description="Message explaining that the Assistant is blocked by the server's remote-access settings"
        />
      </Typography.Text>

      <Button componentId="mlflow.assistant.chat_panel.remote_close" onClick={onClose}>
        <FormattedMessage defaultMessage="Close" description="Button to close the assistant panel on remote servers" />
      </Button>
    </div>
  );
};

/**
 * Setup prompt shown when assistant is not set up yet.
 * Shows empty state illustration and setup button.
 */
const SetupPrompt = ({ onSetup }: { onSetup: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
        padding: theme.spacing.sm,
        gap: theme.spacing.md,
      }}
    >
      <AssistantWelcomeCarousel />

      <Button componentId="mlflow.assistant.chat_panel.setup" type="primary" onClick={onSetup}>
        Get Started
      </Button>
    </div>
  );
};

/**
 * Assistant Chat Panel.
 * Shows the chat header and the chat interface.
 * When setup is incomplete, shows a banner prompting the user to set up.
 */
export const AssistantChatPanel = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const {
    closePanel,
    reset,
    setupComplete,
    isLoadingConfig,
    canUseAssistant,
    completeSetup,
    isLocalServer,
  } = useAssistant();
  const context = useAssistantPageContext();
  const experimentId = context['experimentId'] as string | undefined;

  const [currentView, setCurrentView] = useState<CurrentView>('chat');

  const handleClose = useCallback(() => {
    closePanel();
  }, [closePanel]);

  const handleReset = useCallback(() => {
    reset();
  }, [reset]);

  const handleStartSetup = useCallback(() => {
    setCurrentView('setup-wizard');
  }, []);

  const handleSetupComplete = useCallback(() => {
    setCurrentView('chat');
    completeSetup();
  }, [completeSetup]);

  const handleOpenSettings = useCallback(() => {
    setCurrentView('settings');
  }, []);

  const handleBackFromSettings = useCallback(() => {
    setCurrentView('chat');
  }, []);

  const renderContent = () => {
    // Show message when this client isn't allowed to use the Assistant
    // (e.g. a remote client and the server's remote-access settings don't permit it)
    if (!canUseAssistant) {
      return <RemoteServerMessage onClose={handleClose} />;
    }

    // Show loading state while fetching config
    if (isLoadingConfig) {
      return <SetupLoadingState />;
    }

    switch (currentView) {
      case 'setup-wizard':
        return <AssistantSetupWizard experimentId={experimentId} onComplete={handleSetupComplete} />;
      case 'settings':
        return (
          <AssistantSetupWizard
            experimentId={experimentId}
            onComplete={handleSetupComplete}
            initialStep="provider"
            onBack={handleBackFromSettings}
          />
        );
      case 'chat':
      default:
        // With server-side default resolution this only happens when nothing is
        // usable at all (e.g. a fresh remote-accessible server with no gateway
        // endpoint); the wizard remains as the manual fallback.
        if (!setupComplete) {
          return <SetupPrompt onSetup={handleStartSetup} />;
        }
        return <ChatPanelContent onOpenSettings={handleOpenSettings} />;
    }
  };

  // Determine if we should show the chat controls (new chat button)
  const showChatControls = setupComplete && currentView === 'chat';

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            fontWeight: theme.typography.typographyBoldFontWeight,
            fontSize: theme.typography.fontSizeMd + 2,
          }}
        >
          <SparkleDoubleIcon color="ai" css={{ fontSize: 20 }} />
          <FormattedMessage defaultMessage="MLflow Assistant" description="Title for the global Assistant chat panel" />
          <Tag componentId="mlflow.assistant.chat_panel.beta" color="turquoise">
            Beta
          </Tag>
        </span>
        <div css={{ display: 'flex', gap: theme.spacing.xs }}>
          {showChatControls && (
            <>
              <Tooltip componentId="mlflow.assistant.chat_panel.reset.tooltip" content="New Chat">
                <Button
                  componentId="mlflow.assistant.chat_panel.reset"
                  size="small"
                  icon={<RefreshIcon />}
                  onClick={handleReset}
                  aria-label="New Chat"
                />
              </Tooltip>
              <Tooltip
                componentId="mlflow.assistant.chat_panel.settings.tooltip"
                content={
                  isLocalServer
                    ? intl.formatMessage({
                        defaultMessage: 'Settings',
                        description: 'Tooltip for the Assistant settings button',
                      })
                    : intl.formatMessage({
                        defaultMessage:
                          'Updating Assistant settings is not allowed for a remote MLflow server. ' +
                          'Contact your server admin to update the settings.',
                        description:
                          'Tooltip explaining that Assistant settings cannot be changed from a remote client',
                      })
                }
              >
                <Button
                  componentId="mlflow.assistant.chat_panel.settings"
                  size="small"
                  icon={<GearIcon />}
                  onClick={handleOpenSettings}
                  aria-label="Settings"
                  disabled={!isLocalServer}
                />
              </Tooltip>
            </>
          )}
          <Tooltip componentId="mlflow.assistant.chat_panel.close.tooltip" content="Close">
            <Button
              componentId="mlflow.assistant.chat_panel.close"
              size="small"
              icon={<CloseIcon />}
              onClick={handleClose}
              aria-label="Close"
            />
          </Tooltip>
        </div>
      </div>
      {renderContent()}
    </div>
  );
};
