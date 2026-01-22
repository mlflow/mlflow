/**
 * Assistant Chat Panel component.
 * Displays the chat interface for the Assistant.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Button,
  Card,
  CloseIcon,
  GearIcon,
  RefreshIcon,
  SparkleDoubleIcon,
  SparkleIcon,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  SendIcon,
  WrenchSparkleIcon,
  Tag,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';
import { useAssistantPageContext } from './AssistantPageContext';
import { AssistantContextTags } from './AssistantContextTags';
import type { ChatMessage, ToolUseInfo } from './types';
import { AssistantSetupWizard } from './setup';
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

/**
 * Single chat message bubble.
 */
const ChatMessageBubble = ({
  message,
  isLastMessage,
  activeTools,
  onRegenerate,
}: {
  message: ChatMessage;
  isLastMessage: boolean;
  activeTools?: ToolUseInfo[];
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
          <GenAIMarkdownRenderer>{message.content}</GenAIMarkdownRenderer>
        )}
        {/* Loading indicator */}
        {message.isStreaming && (
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
              css={{
                fontSize: 16,
                animation: 'pulse 1.5s ease-in-out infinite',
                '@keyframes pulse': PULSE_ANIMATION,
              }}
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
              {activeTools && activeTools.length > 0 && activeTools[0].description
                ? `Tool: ${activeTools[0].description}`
                : 'Processing'}
            </span>
          </div>
        )}
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
  const { messages, isStreaming, error, activeTools, sendMessage, regenerateLastMessage } = useAssistant();
  const pageContext = useAssistantPageContext();
  const hasExperimentContext = Boolean(pageContext['experimentId']);

  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
        handleSend();
      }
    },
    [handleSend],
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
              activeTools={message.isStreaming ? activeTools : undefined}
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
          <div css={{ display: 'flex', alignItems: 'center' }}>
            <input
              placeholder="Ask a question..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              css={{
                flex: 1,
                border: 'none',
                outline: 'none',
                backgroundColor: 'transparent',
                fontSize: theme.typography.fontSizeBase,
                color: theme.colors.textPrimary,
                padding: theme.spacing.xs,
                '&::placeholder': {
                  color: theme.colors.textPlaceholder,
                },
                '&:focus': {
                  border: 'none',
                  outline: 'none',
                },
              }}
            />
            <button
              onClick={handleSend}
              disabled={!inputValue.trim() || isStreaming}
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: theme.spacing.xs,
                border: 'none',
                background: 'transparent',
                cursor: !inputValue.trim() || isStreaming ? 'not-allowed' : 'pointer',
                borderRadius: theme.borders.borderRadiusSm,
                color: theme.colors.actionPrimaryBackgroundDefault,
                opacity: !inputValue.trim() || isStreaming ? 0.3 : 1,
                '&:hover:not(:disabled)': {
                  backgroundColor: theme.colors.actionDefaultBackgroundHover,
                },
                '&:active:not(:disabled)': {
                  backgroundColor: theme.colors.actionDefaultBackgroundPress,
                },
              }}
              aria-label="Send message"
            >
              {isStreaming ? <Spinner size="small" /> : <SendIcon />}
            </button>
          </div>
          <AssistantContextTags />
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
            initialStep="project"
            onBack={handleBackFromSettings}
          />
        );
      case 'chat':
      default:
        return setupComplete ? <ChatPanelContent /> : <SetupPrompt onSetup={handleStartSetup} />;
    }
  };

  // Determine if we should show the chat controls (new chat button)
  const showChatControls = Boolean(experimentId) && setupComplete && currentView === 'chat';

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
