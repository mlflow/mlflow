/**
 * Assistant Chat Panel component.
 * Displays the chat interface for the Assistant.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Button,
  Card,
  CloseIcon,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  GearIcon,
  InfoTooltip,
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
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';
import { useAssistantPageContext } from './AssistantPageContext';
import { AssistantContextTags } from './AssistantContextTags';
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

type ToolCallPart = Extract<AssistantPart, { type: 'toolCall' }>;

// One-line, human-readable summary of a tool call's input for the transcript.
const toolInputSummary = (part: ToolCallPart): string => {
  const input = part.input ?? {};
  const command = input['command'];
  const traceId = input['trace_id'];
  const jqFilter = input['jq_filter'];
  const description = input['description'];
  if (typeof command === 'string') return command;
  if (typeof traceId === 'string') {
    return typeof jqFilter === 'string' && jqFilter ? `${traceId} · ${jqFilter}` : traceId;
  }
  if (typeof description === 'string') return description;
  const json = JSON.stringify(input);
  return json === '{}' ? '' : json;
};

/**
 * A persistent record of a tool the assistant called, shown inline in the transcript.
 */
const ToolCallChip = ({ part }: { part: ToolCallPart }) => {
  const { theme } = useDesignSystemTheme();
  const summary = toolInputSummary(part);
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        margin: `${theme.spacing.xs}px 0`,
      }}
      aria-label={`Tool call: ${part.name}`}
    >
      <WrenchSparkleIcon css={{ fontSize: 14, flexShrink: 0, color: theme.colors.textSecondary }} />
      <Typography.Text size="sm" color="secondary" bold>
        {part.name}
      </Typography.Text>
      {summary && (
        <Typography.Text
          size="sm"
          color="secondary"
          css={{
            fontFamily: 'monospace',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {summary}
        </Typography.Text>
      )}
    </div>
  );
};

/**
 * Renders an assistant turn's ordered parts (text + tool calls). Falls back to plain
 * `content` for messages that predate the parts model.
 */
export const AssistantMessageBody = ({ message }: { message: ChatMessage }) => {
  const parts: AssistantPart[] = message.parts ?? [{ type: 'text', text: message.content }];
  return (
    <>
      {parts.map((part, i) =>
        part.type === 'text' ? (
          part.text ? (
            <GenAIMarkdownRenderer key={`text-${i}`}>{part.text}</GenAIMarkdownRenderer>
          ) : null
        ) : (
          <ToolCallChip key={part.toolUseId} part={part} />
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
const ChatPanelContent = () => {
  const { theme } = useDesignSystemTheme();
  const { messages, isStreaming, error, sendMessage, regenerateLastMessage, cancelSession, tokenUsage } =
    useAssistant();
  const logTelemetryEvent = useLogTelemetryEvent();
  const viewId = useMemo(() => uuidv4(), []);

  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

  const handleSend = useCallback(() => {
    if (inputValue.trim() && !isStreaming) {
      sendMessage(inputValue.trim());
      setInputValue('');
    }
  }, [inputValue, isStreaming, sendMessage]);

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
      sendMessage(prompt);
    },
    [sendMessage],
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
          <div css={{ display: 'flex', alignItems: 'flex-end' }}>
            <textarea
              ref={textareaRef}
              placeholder="Ask a question..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              css={{
                flex: 1,
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
            <Button
              componentId="mlflow.assistant.chat_panel.send"
              onClick={isStreaming ? cancelSession : handleSend}
              disabled={!isStreaming && !inputValue.trim()}
              icon={isStreaming ? <StopIcon /> : <SendIcon />}
              aria-label="Send message"
            />
          </div>
          <AssistantContextTags />
          {tokenUsage.totalTokens > 0 && (
            <div
              css={{
                alignSelf: 'flex-end',
                display: 'inline-flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                paddingTop: theme.spacing.sm,
              }}
            >
              <Typography.Text size="sm" color="secondary">
                <FormattedMessage
                  defaultMessage="{total} tokens"
                  description="Compact per-session assistant token count shown near the chat input"
                  values={{ total: formatCompactTokens(tokenUsage.totalTokens) }}
                />
                {tokenUsage.costUsd != null && (
                  <>
                    {' · '}
                    <FormattedMessage
                      defaultMessage="~{cost}"
                      description="Estimated session cost; the leading tilde marks it as an approximate estimate"
                      values={{ cost: formatCostUsd(tokenUsage.costUsd) }}
                    />
                  </>
                )}
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
                    <span>
                      {tokenUsage.costUsd != null ? (
                        <FormattedMessage
                          defaultMessage="Estimated from public model pricing; actual cost may vary (provider and cache rates)."
                          description="Disclaimer clarifying that the displayed assistant cost is an estimate"
                        />
                      ) : (
                        <FormattedMessage
                          defaultMessage="Cost estimate unavailable for this model."
                          description="Shown when the assistant's model is not in the pricing catalog so cost cannot be estimated"
                        />
                      )}
                    </span>
                  </div>
                }
              />
            </div>
          )}
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
 * Message shown when server is not running locally.
 * Assistant only works with local MLflow servers.
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
          description="Title shown when Assistant is not available for remote servers"
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
          defaultMessage="MLflow Assistant is only available when the server is running locally. Remote server support is coming soon."
          description="Message explaining that Assistant only works with local servers"
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
  const { closePanel, reset, setupComplete, isLoadingConfig, isLocalServer, completeSetup } = useAssistant();
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
    // Show message for remote servers - Assistant only works locally
    if (!isLocalServer) {
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
        if (!setupComplete) {
          return <SetupPrompt onSetup={handleStartSetup} />;
        }
        return <ChatPanelContent />;
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
              <Tooltip componentId="mlflow.assistant.chat_panel.settings.tooltip" content="Settings">
                <Button
                  componentId="mlflow.assistant.chat_panel.settings"
                  size="small"
                  icon={<GearIcon />}
                  onClick={handleOpenSettings}
                  aria-label="Settings"
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
