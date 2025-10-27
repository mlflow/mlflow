import { every, isBoolean, isNumber, isString } from 'lodash';
import { useEffect, useMemo, useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerChatToolsRenderer } from './ModelTraceExplorerChatToolsRenderer';
import { ModelTraceExplorerRetrieverFieldRenderer } from './ModelTraceExplorerRetrieverFieldRenderer';
import { ModelTraceExplorerTextFieldRenderer } from './ModelTraceExplorerTextFieldRenderer';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { isModelTraceChatTool, isRetrieverDocument, normalizeConversation } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerConversation } from '../right-pane/ModelTraceExplorerConversation';
import { FormattedMessage } from '@mlflow/mlflow/src/i18n/i18n';

const MAX_VISIBLE_MESSAGES = 3;

export const ModelTraceExplorerFieldRenderer = ({
  title,
  data,
  renderMode,
  chatMessageFormat,
}: {
  title: string;
  data: string;
  renderMode: 'default' | 'json' | 'text';
  chatMessageFormat?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [messagesExpanded, setMessagesExpanded] = useState(false);
  const parsedData = useMemo(() => {
    try {
      return JSON.parse(data);
    } catch (e) {
      return data;
    }
  }, [data]);

  useEffect(() => {
    setMessagesExpanded(false);
  }, [data]);

  const dataIsScalar = isString(parsedData) || isNumber(parsedData) || isBoolean(parsedData);
  // wrap the value in an object with the title as key. this helps normalizeConversation
  // recognize the format, as this util function was designed to receive the whole input
  // object rather than value by value. it does not work for complex cases where we need
  // to check multiple keys in the object (e.g. anthropic), but works for cases where we're
  // basically just looking for the field that contains chat messages.
  const chatMessages = normalizeConversation(title ? { [title]: parsedData } : parsedData, chatMessageFormat);
  const isChatTools = Array.isArray(parsedData) && parsedData.length > 0 && every(parsedData, isModelTraceChatTool);
  const isRetrieverDocuments =
    Array.isArray(parsedData) && parsedData.length > 0 && every(parsedData, isRetrieverDocument);

  if (chatMessages && chatMessages.length > 0) {
    const shouldTruncateMessages = chatMessages.length > MAX_VISIBLE_MESSAGES;
    const visibleMessages =
      messagesExpanded || !shouldTruncateMessages ? chatMessages : chatMessages.slice(-MAX_VISIBLE_MESSAGES);
    const hiddenMessageCount = shouldTruncateMessages ? chatMessages.length - visibleMessages.length : 0;

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          padding: theme.spacing.sm,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusSm,
        }}
      >
        {title && (
          <Typography.Title level={4} color="secondary" withoutMargins css={{ marginLeft: theme.spacing.xs }}>
            {title}
          </Typography.Title>
        )}
        {shouldTruncateMessages && (
          <Typography.Link
            css={{ alignSelf: 'flex-start', marginLeft: theme.spacing.xs }}
            componentId="shared.model-trace-explorer.conversation-toggle"
            onClick={() => setMessagesExpanded(!messagesExpanded)}
          >
            {messagesExpanded ? (
              <FormattedMessage
                defaultMessage="Show less"
                description="Button label to collapse conversation messages in model trace explorer"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Show {hiddenMessageCount} more"
                description="Button label to expand and show hidden conversation messages in model trace explorer"
                values={{ hiddenMessageCount }}
              />
            )}
          </Typography.Link>
        )}
        <ModelTraceExplorerConversation messages={visibleMessages} />
      </div>
    );
  }

  if (renderMode === 'json') {
    return <ModelTraceExplorerCodeSnippet title={title} data={data} initialRenderMode={CodeSnippetRenderMode.JSON} />;
  }

  if (renderMode === 'text') {
    return <ModelTraceExplorerCodeSnippet title={title} data={data} initialRenderMode={CodeSnippetRenderMode.TEXT} />;
  }

  if (dataIsScalar) {
    return <ModelTraceExplorerTextFieldRenderer title={title} value={String(parsedData)} />;
  }

  if (isChatTools) {
    return <ModelTraceExplorerChatToolsRenderer title={title} tools={parsedData} />;
  }

  if (isRetrieverDocuments) {
    return <ModelTraceExplorerRetrieverFieldRenderer title={title} documents={parsedData} />;
  }

  return <ModelTraceExplorerCodeSnippet title={title} data={data} />;
};
