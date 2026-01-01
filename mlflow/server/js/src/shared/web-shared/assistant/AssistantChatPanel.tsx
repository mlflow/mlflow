/**
 * Assistant Chat Panel component.
 * Displays the chat interface for the Assistant.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Button,
  Card,
  CloseIcon,
  CopyIcon,
  PlusIcon,
  RefreshIcon,
  SparkleDoubleIcon,
  SparkleIcon,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  SendIcon,
  WrenchSparkleIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';
import type { ChatMessage } from './types';
import { GenAIMarkdownRenderer } from '../genai-markdown-renderer';

const COMPONENT_ID = 'mlflow.assistant.chat_panel';

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
const ChatMessageBubble = ({ message, isLastMessage }: { message: ChatMessage; isLastMessage: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const isUser = message.role === 'user';
  const [isHovered, setIsHovered] = useState(false);

  const handleCopy = () => {
    // TODO: Implement copy functionality
  };

  const handleRegenerate = () => {
    // TODO: Implement regenerate functionality
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
          padding: `${theme.spacing.md}px ${theme.spacing.md}px ${isUser ? theme.spacing.md : 0}px ${theme.spacing.md}px`,
          borderRadius: theme.borders.borderRadiusLg,
          backgroundColor: isUser ? theme.colors.blue100 : 'transparent',
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
              Processing
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
          <Tooltip componentId={`${COMPONENT_ID}.copy.tooltip`} content="Copy">
            <Button
              componentId={`${COMPONENT_ID}.copy`}
              size="small"
              icon={<CopyIcon />}
              onClick={handleCopy}
            />
          </Tooltip>
          {isLastMessage && (
            <Tooltip componentId={`${COMPONENT_ID}.regenerate.tooltip`} content="Regenerate">
              <Button
                componentId={`${COMPONENT_ID}.regenerate`}
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

  const suggestions = [
    'What does this trace show?',
    'Debug the error in this trace.',
    'What is the performance bottleneck in this trace?',
  ];

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
            componentId={`${COMPONENT_ID}.suggestion.card`}
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
 * Status indicator showing processing state.
 */
const StatusIndicator = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        color: theme.colors.textSecondary,
      }}
    >
      <SparkleIcon
        color="ai"
        css={{
          fontSize: 18,
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
        Processing
      </span>
    </div>
  );
};

/**
 * Chat panel content component.
 */
const ChatPanelContent = () => {
  const { theme } = useDesignSystemTheme();
  const { messages, isStreaming, error, currentStatus, sendMessage } = useAssistant();

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
        height: '100%',
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
          return <ChatMessageBubble key={message.id} message={message} isLastMessage={isLastAssistantMessage} />;
        })}

        <div ref={messagesEndRef} />
      </div>

      {/* Status indicator */}
      {currentStatus && <StatusIndicator />}

      {/* Input area */}
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px ${theme.spacing.md}px ${theme.spacing.md}px`,
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          <input
            placeholder="Ask a question about this trace..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
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
              '&:disabled': {
                cursor: 'not-allowed',
                opacity: 0.5,
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
      </div>
    </div>
  );
};

/**
 * Assistant Chat Panel.
 * Shows the chat header and the chat interface.
 */
export const AssistantChatPanel = () => {
  const { theme } = useDesignSystemTheme();
  const { closePanel, reset } = useAssistant();

  const handleClose = () => {
    closePanel();
  };

  const handleReset = () => {
    reset();
  };

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
        </span>
        <div css={{ display: 'flex', gap: theme.spacing.xs }}>
          <Tooltip componentId={`${COMPONENT_ID}.reset.tooltip`} content="New Chat">
            <Button
              componentId={`${COMPONENT_ID}.reset`}
              size="small"
              icon={<PlusIcon />}
              onClick={handleReset}
              aria-label="New Chat"
            />
          </Tooltip>
          <Tooltip componentId={`${COMPONENT_ID}.close.tooltip`} content="Close">
            <Button
              componentId={`${COMPONENT_ID}.close`}
              size="small"
              icon={<CloseIcon />}
              onClick={handleClose}
              aria-label="Close"
            />
          </Tooltip>
        </div>
      </div>
      <ChatPanelContent />
    </div>
  );
};
