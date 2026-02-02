import { isNil } from 'lodash';
import { useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  ChevronUpIcon,
  LightbulbIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';

import { ModelTraceExplorerChatMessageHeader } from './ModelTraceExplorerChatMessageHeader';
import { CONTENT_TRUNCATION_LIMIT } from './ModelTraceExplorerChatRenderer.utils';
import { ModelTraceExplorerToolCallMessage } from './ModelTraceExplorerToolCallMessage';
import { CodeSnippetRenderMode, type ModelTraceChatMessage } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from '../ModelTraceExplorerCodeSnippetBody';

const tryGetJsonContent = (content: string) => {
  try {
    return {
      content: JSON.stringify(JSON.parse(content), null, 2),
      isJson: true,
    };
  } catch (error) {
    return {
      content,
      isJson: false,
    };
  }
};

function ModelTraceExplorerChatMessageContent({
  content,
  shouldDisplayCodeSnippet,
}: {
  content: string;
  shouldDisplayCodeSnippet: boolean;
}) {
  const { theme } = useDesignSystemTheme();

  if (!content) {
    return null;
  }

  if (shouldDisplayCodeSnippet) {
    return (
      <ModelTraceExplorerCodeSnippetBody
        data={content}
        searchFilter=""
        activeMatch={null}
        containsActiveMatch={false}
        renderMode={CodeSnippetRenderMode.JSON}
      />
    );
  }

  return (
    <div
      css={{
        padding: theme.spacing.sm,
        paddingTop: 0,
        // genai markdown renderer uses default paragraph sizing which has
        // a bottom margin that we can't get rid of. workaround by setting
        // negative margin in a wrapper.
        marginBottom: -theme.typography.fontSizeBase,
      }}
    >
      <GenAIMarkdownRenderer>{content}</GenAIMarkdownRenderer>
    </div>
  );
}

function ModelTraceExplorerReasoningSection({ reasoning }: { reasoning: string }) {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      css={{
        margin: theme.spacing.sm,
        marginTop: 0,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
        overflow: 'hidden',
      }}
    >
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          width: '100%',
          padding: theme.spacing.sm,
          backgroundColor: 'transparent',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
          '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
          },
        }}
      >
        {expanded ? (
          <ChevronDownIcon css={{ color: theme.colors.textSecondary }} />
        ) : (
          <ChevronRightIcon css={{ color: theme.colors.textSecondary }} />
        )}
        <LightbulbIcon css={{ color: theme.colors.textValidationWarning }} />
        <Typography.Text bold css={{ color: theme.colors.textSecondary }}>
          <FormattedMessage
            defaultMessage="Reasoning"
            description="Label for the collapsible reasoning section in chat message"
          />
        </Typography.Text>
      </button>
      {expanded && (
        <div
          css={{
            padding: theme.spacing.sm,
            paddingTop: 0,
            borderTop: `1px solid ${theme.colors.borderDecorative}`,
            marginBottom: -theme.typography.fontSizeBase,
          }}
        >
          <GenAIMarkdownRenderer>{reasoning}</GenAIMarkdownRenderer>
        </div>
      )}
    </div>
  );
}

export function ModelTraceExplorerChatMessage({
  message,
  className,
}: {
  message: ModelTraceChatMessage;
  className?: string;
}) {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const { content, isJson } = tryGetJsonContent(message.content ?? '');

  // tool call responses can be JSON, and in these cases
  // it's more helpful to display the message as JSON
  const shouldDisplayCodeSnippet = isJson && (message.role === 'tool' || message.role === 'function');
  // if the content is JSON, truncation will be handled by the code
  // snippet. otherwise, we need to truncate the content manually.
  const isExpandable = !shouldDisplayCodeSnippet && content.length > CONTENT_TRUNCATION_LIMIT;

  const displayedContent = isExpandable && !expanded ? `${content.slice(0, CONTENT_TRUNCATION_LIMIT)}...` : content;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        borderRadius: theme.borders.borderRadiusSm,
        border: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundPrimary,
        overflow: 'hidden',
      }}
      className={className}
    >
      <ModelTraceExplorerChatMessageHeader
        isExpandable={isExpandable}
        expanded={expanded}
        setExpanded={setExpanded}
        message={message}
      />
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {message.reasoning && <ModelTraceExplorerReasoningSection reasoning={message.reasoning} />}
        {!isNil(message.tool_calls) &&
          message.tool_calls.map((toolCall) => (
            <ModelTraceExplorerToolCallMessage key={toolCall.id} toolCall={toolCall} />
          ))}
        <ModelTraceExplorerChatMessageContent
          content={displayedContent}
          shouldDisplayCodeSnippet={shouldDisplayCodeSnippet}
        />
      </div>
      {isExpandable && (
        <Button
          componentId={
            expanded
              ? 'shared.model-trace-explorer.chat-message-see-less'
              : 'shared.model-trace-explorer.chat-message-see-more'
          }
          icon={expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
          type="tertiary"
          onClick={() => setExpanded(!expanded)}
          css={{
            display: 'flex',
            width: '100%',
            padding: theme.spacing.md,
            borderRadius: '0px !important',
          }}
        >
          {expanded ? (
            <FormattedMessage
              defaultMessage="See less"
              description="A button label in a message renderer that truncates long content when clicked."
            />
          ) : (
            <FormattedMessage
              defaultMessage="See more"
              description="A button label in a message renderer that expands truncated content when clicked."
            />
          )}
        </Button>
      )}
    </div>
  );
}
