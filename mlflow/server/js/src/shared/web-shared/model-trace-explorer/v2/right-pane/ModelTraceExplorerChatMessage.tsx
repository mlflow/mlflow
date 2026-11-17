import { memo, useMemo, useState } from 'react';

import {
  ChevronDownIcon,
  ChevronRightIcon,
  LightbulbIcon,
  Modal,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '../../../genai-markdown-renderer/GenAIMarkdownRenderer';
import { DownloadLink, exceedsRenderSizeLimit } from '../../../media-rendering-utils';
import {
  attachmentAwareUrlTransform,
  isAttachmentUri,
  parseAttachmentUri,
  useAttachmentUrl,
} from '../../attachment-utils';
import { ModelTraceExplorerAttachmentRenderer } from '../field-renderers/ModelTraceExplorerAttachmentRenderer';

import { ModelTraceExplorerChatMessageHeader } from './ModelTraceExplorerChatMessageHeader';
import {
  CONTENT_TRUNCATION_LIMIT,
  getDisplayLength,
  truncatePreservingImages,
} from './ModelTraceExplorerChatRenderer.utils';
import {
  ModelTraceExplorerToolCallIdLink,
  ModelTraceExplorerToolCallMessage,
  ModelTraceExplorerToolIcon,
} from './ModelTraceExplorerToolCallMessage';
import { CodeSnippetRenderMode, type ModelTraceChatMessage, type ModelTraceInputAudio } from '../ModelTrace.types';
import { LARGE_TRACE_TEXT_LIMIT, MARKDOWN_RENDER_SIZE_LIMIT } from '../../constants';
import { ModelTraceExplorerCodeSnippetBody } from '../ModelTraceExplorerCodeSnippetBody';
import { ModelTraceExplorerLargeText } from '../ModelTraceExplorerLargeText';

function ClickToExpandImage({ src, alt }: { src: string; alt?: string }) {
  const { theme } = useDesignSystemTheme();
  const [previewVisible, setPreviewVisible] = useState(false);
  return (
    <>
      <img
        src={src}
        alt={alt}
        css={{
          maxWidth: '100%',
          maxHeight: 200,
          cursor: 'pointer',
          '&:hover': { boxShadow: `0 0 4px ${theme.colors.border}` },
        }}
        onClick={() => setPreviewVisible(true)}
      />
      <Modal
        componentId="shared.model-trace-explorer.image-preview"
        title=""
        visible={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        onOk={() => setPreviewVisible(false)}
      >
        <img src={src} alt={alt} css={{ maxWidth: '100%', maxHeight: '70vh', display: 'block' }} />
      </Modal>
    </>
  );
}

function AttachmentImage({ src, alt }: { src?: string; alt?: string }) {
  const { url, contentLength, contentType, loading, error, triggerDownload } = useAttachmentUrl(src ?? null);
  if (loading) {
    return <Spinner size="small" />;
  }
  if ((url || triggerDownload) && contentType && exceedsRenderSizeLimit(contentType, contentLength)) {
    return (
      <DownloadLink
        url={url}
        contentType={contentType}
        contentLength={contentLength}
        onFetchDownload={triggerDownload}
      />
    );
  }
  if (error || !url) {
    return <span>{`[${alt ?? 'Failed to load image'}]`}</span>;
  }
  return <ClickToExpandImage src={url} alt={alt} />;
}

const attachmentAwareImgRenderer = ({ src, alt }: { src?: string; alt?: string }) => {
  if (src && isAttachmentUri(src)) {
    const parsed = parseAttachmentUri(src);
    if (parsed && !parsed.contentType.startsWith('image/')) {
      // Non-image attachments (PDF, audio, etc.) use the full AttachmentRenderer
      return (
        <ModelTraceExplorerAttachmentRenderer
          title=""
          attachmentId={parsed.attachmentId}
          traceId={parsed.traceId}
          contentType={parsed.contentType}
          size={parsed.size}
        />
      );
    }
    return <AttachmentImage src={src} alt={alt} />;
  }
  if (src) {
    return <ClickToExpandImage src={src} alt={alt} />;
  }
  return <img src={src} alt={alt} css={{ maxWidth: '100%' }} />;
};

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

const ChatMarkdown = memo(GenAIMarkdownRenderer);
const CHAT_MARKDOWN_COMPONENTS = { img: attachmentAwareImgRenderer };

function ModelTraceExplorerChatMessageContent({
  content,
  shouldDisplayCodeSnippet,
  showsLargeText = false,
  copyText,
}: {
  content: string;
  shouldDisplayCodeSnippet: boolean;
  showsLargeText?: boolean;
  copyText?: string;
}): React.ReactElement | null {
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

  if (showsLargeText) {
    return (
      <div css={{ padding: theme.spacing.sm, paddingTop: 0 }}>
        <ModelTraceExplorerLargeText text={content} copyText={copyText} />
      </div>
    );
  }

  // Image URLs do not count toward the text limit; retain the existing cap for image-heavy messages.
  if (content.length > MARKDOWN_RENDER_SIZE_LIMIT) {
    return (
      <div css={{ padding: theme.spacing.sm, paddingTop: 0 }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Content too large to render ({size}). Displaying as plain text."
            description="Message shown when chat content exceeds the markdown rendering size limit"
            values={{ size: `${(content.length / 1_000_000).toFixed(1)}MB` }}
          />
        </Typography.Text>
        <pre
          css={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
            maxHeight: 400,
            overflow: 'auto',
            fontSize: 12,
          }}
        >
          {content.slice(0, 10_000)}
        </pre>
      </div>
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
      <ChatMarkdown components={CHAT_MARKDOWN_COMPONENTS} urlTransform={attachmentAwareUrlTransform}>
        {content}
      </ChatMarkdown>
    </div>
  );
}

function ModelTraceExplorerReasoningSection({ reasoning }: { reasoning: string }) {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const isLargeReasoning =
    reasoning.length > LARGE_TRACE_TEXT_LIMIT &&
    getDisplayLength(reasoning) === reasoning.length &&
    !reasoning.includes('mlflow-attachment://');

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
            marginBottom: isLargeReasoning ? 0 : -theme.typography.fontSizeBase,
          }}
        >
          {isLargeReasoning ? (
            <ModelTraceExplorerLargeText text={reasoning} copyText={reasoning} />
          ) : (
            <ChatMarkdown>{reasoning}</ChatMarkdown>
          )}
        </div>
      )}
    </div>
  );
}

function getAudioMimeType(format: string): string {
  switch (format) {
    case 'mp3':
      return 'audio/mpeg';
    default:
      return `audio/${format}`;
  }
}

function AttachmentAudioPlayer({ uri }: { uri: string }) {
  const { url, contentLength, contentType, loading, error, triggerDownload } = useAttachmentUrl(uri);
  if (loading) {
    return <Spinner size="small" />;
  }
  if ((url || triggerDownload) && contentType && exceedsRenderSizeLimit(contentType, contentLength)) {
    return (
      <DownloadLink
        url={url}
        contentType={contentType}
        contentLength={contentLength}
        onFetchDownload={triggerDownload}
      />
    );
  }
  if (error || !url) {
    return (
      <Typography.Text color="error">
        <FormattedMessage
          defaultMessage="Failed to load audio attachment"
          description="Error message when trace audio attachment fails to load"
        />
      </Typography.Text>
    );
  }
  // eslint-disable-next-line jsx-a11y/media-has-caption
  return <audio controls css={{ width: '100%', maxWidth: 500 }} src={url} />;
}

function ModelTraceExplorerAudioPlayer({ audioParts }: { audioParts: ModelTraceInputAudio[] }) {
  const { theme } = useDesignSystemTheme();

  return (
    <>
      {audioParts.map((audio, index) => (
        <div
          key={index}
          css={{
            padding: theme.spacing.sm,
            paddingTop: 0,
          }}
        >
          {isAttachmentUri(audio.data) ? (
            <AttachmentAudioPlayer uri={audio.data} />
          ) : (
            // eslint-disable-next-line jsx-a11y/media-has-caption
            <audio
              controls
              css={{ width: '100%', maxWidth: 500 }}
              src={`data:${getAudioMimeType(audio.format)};base64,${audio.data}`}
            />
          )}
        </div>
      ))}
    </>
  );
}

export function ModelTraceExplorerChatMessage({
  message,
  className,
  toolCallName,
}: {
  message: ModelTraceChatMessage;
  className?: string;
  toolCallName?: string;
}): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const { content, isJson, attachmentRefCount, displayLength } = useMemo(() => {
    const parsed = tryGetJsonContent(message.content ?? '');
    return {
      ...parsed,
      attachmentRefCount: parsed.content.split('mlflow-attachment://').length - 1,
      displayLength: getDisplayLength(parsed.content),
    };
  }, [message.content]);

  // tool call responses can be JSON, and in these cases
  // it's more helpful to display the message as JSON
  const shouldDisplayCodeSnippet = isJson && (message.role === 'tool' || message.role === 'function');
  // if the content is JSON, truncation will be handled by the code
  // snippet. otherwise, we need to truncate the content manually.
  // Increase truncation limit when attachment refs are present since the refs themselves
  // are lightweight (~120 chars each) and truncating mid-ref breaks markdown rendering.
  const effectiveLimit = CONTENT_TRUNCATION_LIMIT + attachmentRefCount * 150;
  const isExpandable = !shouldDisplayCodeSnippet && displayLength > effectiveLimit;

  const displayedContent = useMemo(
    () => (isExpandable && !expanded ? truncatePreservingImages(content, effectiveLimit) : content),
    [content, effectiveLimit, expanded, isExpandable],
  );
  // Mixed media must keep its image/attachment-aware Markdown path.
  const showsLargeText =
    !shouldDisplayCodeSnippet &&
    attachmentRefCount === 0 &&
    displayLength === content.length &&
    displayedContent.length > LARGE_TRACE_TEXT_LIMIT;
  const toolCallId = message.tool_call_id;
  const isToolResponse = (message.role === 'tool' || message.role === 'function') && toolCallId;

  if (isToolResponse && toolCallId) {
    const displayName =
      toolCallName ??
      message.name ??
      (message.role === 'function' ? (
        <FormattedMessage
          defaultMessage="Function"
          description="Fallback name for a function response in a model trace"
        />
      ) : (
        <FormattedMessage defaultMessage="Tool" description="Fallback name for a tool response in a model trace" />
      ));

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          backgroundColor: theme.colors.backgroundPrimary,
          overflow: 'hidden',
        }}
        className={className}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            padding: theme.spacing.sm,
            minWidth: 0,
          }}
        >
          <ModelTraceExplorerToolIcon />
          <Typography.Text bold css={{ whiteSpace: 'nowrap', flexShrink: 0 }}>
            {displayName}
          </Typography.Text>
          <ModelTraceExplorerToolCallIdLink toolCallId={toolCallId} />
          {displayedContent && !shouldDisplayCodeSnippet && !showsLargeText && (
            <Typography.Text color="secondary" css={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', minWidth: 0 }}>
              {displayedContent}
            </Typography.Text>
          )}
        </div>
        {(showsLargeText || shouldDisplayCodeSnippet) && (
          <ModelTraceExplorerChatMessageContent
            content={displayedContent}
            shouldDisplayCodeSnippet={shouldDisplayCodeSnippet}
            showsLargeText={showsLargeText}
            copyText={content}
          />
        )}
        {isExpandable && (
          <Typography.Link
            componentId={
              expanded
                ? 'shared.model-trace-explorer.chat-message-see-less'
                : 'shared.model-trace-explorer.chat-message-see-more'
            }
            onClick={() => setExpanded(!expanded)}
            css={{
              padding: theme.spacing.sm,
              paddingTop: 0,
              display: 'flex',
              alignItems: 'center',
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
          </Typography.Link>
        )}
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
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
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.md,
              paddingLeft: theme.spacing.lg,
              paddingRight: theme.spacing.sm,
              paddingBottom: theme.spacing.sm,
            }}
          >
            {message.tool_calls.map((toolCall) => (
              <ModelTraceExplorerToolCallMessage key={toolCall.id} toolCall={toolCall} />
            ))}
          </div>
        )}
        {/* Text content renders before audio parts. The markdown renderer and audio
            player are separate rendering paths, so original part interleaving is not
            preserved. Text-first matches the typical pattern where prompts precede
            media (see https://developers.openai.com/api/docs/guides/audio). */}
        <ModelTraceExplorerChatMessageContent
          content={displayedContent}
          shouldDisplayCodeSnippet={shouldDisplayCodeSnippet}
          showsLargeText={showsLargeText}
          copyText={content}
        />
        {message.audioParts && message.audioParts.length > 0 && (
          <ModelTraceExplorerAudioPlayer audioParts={message.audioParts} />
        )}
      </div>
      {isExpandable && (
        <Typography.Link
          componentId={
            expanded
              ? 'shared.model-trace-explorer.chat-message-see-less'
              : 'shared.model-trace-explorer.chat-message-see-more'
          }
          onClick={() => setExpanded(!expanded)}
          css={{
            padding: theme.spacing.sm,
            display: 'flex',
            alignItems: 'center',
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
        </Typography.Link>
      )}
    </div>
  );
}
