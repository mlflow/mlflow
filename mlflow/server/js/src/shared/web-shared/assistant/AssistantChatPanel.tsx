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
  ForkHorizontalIcon,
  FunctionIcon,
  PlayIcon,
  PlusIcon,
  RefreshIcon,
  SparkleDoubleIcon,
  SparkleIcon,
  Spinner,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  SendIcon,
  WrenchSparkleIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';
import { useAssistantPageContextValue } from './AssistantPageContext';
import type { ChatMessage, ToolUseInfo } from './types';
import { GenAIMarkdownRenderer } from '../genai-markdown-renderer';
import { AssistantSetupWizard } from './setup';

const COMPONENT_ID = 'mlflow.assistant.chat_panel';

// Shared animation keyframes
const PULSE_ANIMATION = {
  '0%, 100%': { transform: 'scale(1)' },
  '50%': { transform: 'scale(1.3)' },
};

const BLINK_ANIMATION = {
  '0%, 100%': { opacity: 1 },
  '50%': { opacity: 0.6 },
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
          padding: `${theme.spacing.md}px ${theme.spacing.md}px ${isUser ? theme.spacing.md : 0}px ${
            theme.spacing.md
          }px`,
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
            <Button componentId={`${COMPONENT_ID}.copy`} size="small" icon={<CopyIcon />} onClick={handleCopy} />
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
const PromptSuggestions = ({
  onSelect,
  disabled = false,
}: {
  onSelect: (prompt: string) => void;
  disabled?: boolean;
}) => {
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
          opacity: disabled ? 0.5 : 1,
        }}
      >
        {suggestions.map((suggestion) => (
          <Card
            componentId={`${COMPONENT_ID}.suggestion.card`}
            key={suggestion}
            onClick={disabled ? undefined : () => onSelect(suggestion)}
            css={{
              cursor: disabled ? 'not-allowed' : 'pointer',
              padding: theme.spacing.sm,
              transition: 'all 0.2s ease',
              '&:hover': disabled
                ? {}
                : {
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
        flexShrink: 0,
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
 * Format tool input arguments for display with truncation.
 */
const formatToolInput = (input?: Record<string, any>, maxLength: number = 80): string => {
  if (!input) return '';

  // Create a simplified display of the input arguments
  const entries = Object.entries(input)
    .filter(([key]) => key !== 'description') // Exclude description since we show it separately
    .map(([key, value]) => {
      if (typeof value === 'string') {
        return `${key}="${value}"`;
      }
      return `${key}=${JSON.stringify(value)}`;
    });

  if (entries.length === 0) return '';

  const fullText = entries.join(', ');
  if (fullText.length <= maxLength) {
    return fullText;
  }

  return fullText.substring(0, maxLength - 3) + '...';
};

/**
 * Tool usage message showing active tools inline in chat history.
 */
const ToolUsageMessage = ({ tools }: { tools: ToolUseInfo[] }) => {
  const { theme } = useDesignSystemTheme();
  const [expandedToolId, setExpandedToolId] = useState<string | null>(null);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        marginBottom: 0,
      }}
    >
      <div
        css={{
          maxWidth: '100%',
          padding: `${theme.spacing.md}px ${theme.spacing.md}px 0px ${theme.spacing.md}px`,
          color: theme.colors.textSecondary,
        }}
      >
        {tools.map((tool) => {
          const inputStr = formatToolInput(tool.input);
          const isExpanded = expandedToolId === tool.id;
          const displayText = tool.description || `Running tool ${tool.name}`;

          return (
            <div
              key={tool.id}
              css={{
                marginBottom: theme.spacing.sm,
                animation: 'blink 1.5s ease-in-out infinite',
                '@keyframes blink': BLINK_ANIMATION,
              }}
            >
              {/* First line: description with expand icon */}
              <div
                css={{
                  display: 'grid',
                  gridTemplateColumns: 'auto auto 1fr',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  fontSize: theme.typography.fontSizeBase,
                }}
              >
                <span
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedToolId(isExpanded ? null : tool.id);
                  }}
                  css={{
                    cursor: 'pointer',
                    padding: theme.spacing.xs,
                    display: 'flex',
                    alignItems: 'center',
                    '&:hover': {
                      opacity: 0.7,
                    },
                  }}
                >
                  {isExpanded ? '▼' : '▶'}
                </span>
                <FunctionIcon
                  css={{
                    fontSize: 18,
                  }}
                />
                <span
                  css={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    '&::after': {
                      content: '"..."',
                      animation: 'dots 1.5s steps(3, end) infinite',
                      display: 'inline-block',
                      width: '1.2em',
                    },
                    '@keyframes dots': DOTS_ANIMATION,
                  }}
                >
                  {displayText}
                </span>
              </div>

              {/* Second line: tool name and inputs (when expanded) */}
              {isExpanded && (
                <div
                  css={{
                    marginTop: theme.spacing.xs,
                    fontSize: theme.typography.fontSizeSm,
                    color: theme.colors.textSecondary,
                    opacity: 0.8,
                  }}
                >
                  <Typography.Text
                    code
                    css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}
                  >
                    {tool.name}
                  </Typography.Text>
                  {inputStr && (
                    <>
                      {' '}
                      <span css={{ fontStyle: 'italic' }}>({inputStr})</span>
                    </>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

/**
 * Context tags showing current page context (compact version for input area).
 */
const AssistantContextTags = () => {
  const { theme } = useDesignSystemTheme();
  const context = useAssistantPageContextValue();

  const traceId = context['traceId'] as string | undefined;
  const selectedTraceIds = context['selectedTraceIds'] as string[] | undefined;
  const runId = context['runId'] as string | undefined;
  const selectedRunIds = context['selectedRunIds'] as string[] | undefined;

  // Dedupe: exclude the currently open item from selected lists
  const filteredSelectedTraceIds = selectedTraceIds?.filter((id) => id !== traceId);
  const filteredSelectedRunIds = selectedRunIds?.filter((id) => id !== runId);

  // Don't render if no context
  if (!traceId && !filteredSelectedTraceIds?.length && !runId && !filteredSelectedRunIds?.length) {
    return null;
  }

  const truncate = (id: string, maxLen = 8) => (id.length > maxLen ? `${id.slice(0, maxLen)}...` : id);

  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs, paddingTop: theme.spacing.sm }}>
      {/* Trace context */}
      {traceId && (
        <Tooltip componentId={`${COMPONENT_ID}.context.trace.tooltip`} content={`Trace: ${traceId}`}>
          <Tag componentId={`${COMPONENT_ID}.context.trace`} color="indigo">
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <ForkHorizontalIcon css={{ fontSize: 12 }} />
              <span>{truncate(traceId)}</span>
            </span>
          </Tag>
        </Tooltip>
      )}
      {filteredSelectedTraceIds && filteredSelectedTraceIds.length > 0 && (
        <>
          {filteredSelectedTraceIds.slice(0, 3).map((id) => (
            <Tooltip key={id} componentId={`${COMPONENT_ID}.context.selected-trace.tooltip`} content={`Trace: ${id}`}>
              <Tag componentId={`${COMPONENT_ID}.context.selected-trace`} color="indigo">
                <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ForkHorizontalIcon css={{ fontSize: 12 }} />
                  <span>{truncate(id)}</span>
                </span>
              </Tag>
            </Tooltip>
          ))}
          {filteredSelectedTraceIds.length > 3 && (
            <Tooltip
              componentId={`${COMPONENT_ID}.context.more-traces.tooltip`}
              content={`${filteredSelectedTraceIds.length - 3} more traces selected`}
            >
              <Tag componentId={`${COMPONENT_ID}.context.more-traces`} color="indigo">
                +{filteredSelectedTraceIds.length - 3}
              </Tag>
            </Tooltip>
          )}
        </>
      )}
      {/* Run context */}
      {runId && (
        <Tooltip componentId={`${COMPONENT_ID}.context.run.tooltip`} content={`Run: ${runId}`}>
          <Tag componentId={`${COMPONENT_ID}.context.run`} color="turquoise">
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <PlayIcon css={{ fontSize: 12 }} />
              <span>{truncate(runId)}</span>
            </span>
          </Tag>
        </Tooltip>
      )}
      {filteredSelectedRunIds && filteredSelectedRunIds.length > 0 && (
        <>
          {filteredSelectedRunIds.slice(0, 3).map((id) => (
            <Tooltip key={id} componentId={`${COMPONENT_ID}.context.selected-run.tooltip`} content={`Run: ${id}`}>
              <Tag componentId={`${COMPONENT_ID}.context.selected-run`} color="turquoise">
                <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <PlayIcon css={{ fontSize: 12 }} />
                  <span>{truncate(id)}</span>
                </span>
              </Tag>
            </Tooltip>
          ))}
          {filteredSelectedRunIds.length > 3 && (
            <Tooltip
              componentId={`${COMPONENT_ID}.context.more-runs.tooltip`}
              content={`${filteredSelectedRunIds.length - 3} more runs selected`}
            >
              <Tag componentId={`${COMPONENT_ID}.context.more-runs`} color="turquoise">
                +{filteredSelectedRunIds.length - 3}
              </Tag>
            </Tooltip>
          )}
        </>
      )}
    </div>
  );
};

interface ChatPanelContentProps {
  disabled?: boolean;
}

/**
 * Chat panel content component.
 */
const ChatPanelContent = ({ disabled = false }: ChatPanelContentProps) => {
  const { theme } = useDesignSystemTheme();
  const { messages, isStreaming, error, currentStatus, activeTools, sendMessage } = useAssistant();

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
        {messages.length === 0 && !isStreaming && (
          <PromptSuggestions onSelect={handleSuggestionSelect} disabled={disabled} />
        )}

        {messages.map((message, index) => {
          // Check if this is the last assistant message
          const isLastAssistantMessage = message.role === 'assistant' && index === messages.length - 1;
          return <ChatMessageBubble key={message.id} message={message} isLastMessage={isLastAssistantMessage} />;
        })}

        {/* Show active tools inline in message history */}
        {activeTools.length > 0 && <ToolUsageMessage tools={activeTools} />}

        <div ref={messagesEndRef} />
      </div>

      {/* Status indicator */}
      {currentStatus && <StatusIndicator />}

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
              placeholder="Ask a question about this trace..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming || disabled}
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
              disabled={!inputValue.trim() || isStreaming || disabled}
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: theme.spacing.xs,
                border: 'none',
                background: 'transparent',
                cursor: !inputValue.trim() || isStreaming || disabled ? 'not-allowed' : 'pointer',
                borderRadius: theme.borders.borderRadiusSm,
                color: theme.colors.actionPrimaryBackgroundDefault,
                opacity: !inputValue.trim() || isStreaming || disabled ? 0.3 : 1,
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
 * Banner shown when assistant is not set up yet.
 */
const SetupBanner = ({ onSetup }: { onSetup: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        margin: theme.spacing.md,
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
      }}
    >
      <SparkleDoubleIcon color="ai" css={{ fontSize: 32 }} />
      <div css={{ textAlign: 'center' }}>
        <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
          Welcome to MLflow Assistant!
        </Typography.Text>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          Complete a quick setup to start chatting.
        </Typography.Text>
      </div>
      <Button componentId={`${COMPONENT_ID}.setup`} type="primary" onClick={onSetup}>
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
  const { closePanel, reset, setupComplete, isLoadingConfig, completeSetup } = useAssistant();
  const context = useAssistantPageContextValue();
  const experimentId = context['experimentId'] as string | undefined;

  // Track whether the user is in the setup wizard
  const [isInSetupWizard, setIsInSetupWizard] = useState(false);

  const handleClose = () => {
    closePanel();
  };

  const handleReset = () => {
    reset();
  };

  const handleStartSetup = () => {
    setIsInSetupWizard(true);
  };

  const handleSetupComplete = () => {
    setIsInSetupWizard(false);
    completeSetup();
  };

  // Determine if setup is needed
  const needsSetup = !setupComplete;

  // Determine what to show in the content area
  const renderContent = () => {
    // Show loading state while fetching config
    if (isLoadingConfig) {
      return <SetupLoadingState />;
    }

    // Show setup wizard if user clicked "Setup"
    if (isInSetupWizard) {
      return <AssistantSetupWizard experimentId={experimentId} onComplete={handleSetupComplete} />;
    }

    // Show chat panel content (with disabled input if setup incomplete)
    return <ChatPanelContent disabled={needsSetup} />;
  };

  // Determine if we should show the chat controls (new chat button)
  const showChatControls = setupComplete && !isInSetupWizard;

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
          <Tag componentId={`${COMPONENT_ID}.beta`} color="turquoise">
            Beta
          </Tag>
        </span>
        <div css={{ display: 'flex', gap: theme.spacing.xs }}>
          {showChatControls && (
            <Tooltip componentId={`${COMPONENT_ID}.reset.tooltip`} content="New Chat">
              <Button
                componentId={`${COMPONENT_ID}.reset`}
                size="small"
                icon={<PlusIcon />}
                onClick={handleReset}
                aria-label="New Chat"
              />
            </Tooltip>
          )}
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
      {needsSetup && !isInSetupWizard && <SetupBanner onSetup={handleStartSetup} />}
      {renderContent()}
    </div>
  );
};
