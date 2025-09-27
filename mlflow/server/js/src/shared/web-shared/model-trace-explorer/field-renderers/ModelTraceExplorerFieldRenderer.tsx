import { every, isString } from 'lodash';
import { useMemo } from 'react';

import { ModelTraceExplorerChatToolsRenderer } from './ModelTraceExplorerChatToolsRenderer';
import { ModelTraceExplorerRetrieverFieldRenderer } from './ModelTraceExplorerRetrieverFieldRenderer';
import { ModelTraceExplorerTextFieldRenderer } from './ModelTraceExplorerTextFieldRenderer';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { isModelTraceChatTool, isRetrieverDocument, normalizeConversation } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerAttachmentRenderer } from './ModelTraceExplorerAttachmentRenderer';
import { ModelTraceExplorerConversation } from '../right-pane/ModelTraceExplorerConversation';

function parseAttachmentUri(uri: string): { id: string; traceId: string; contentType: string } | null {
  // uri looks like "attachment://<id>?content_type=<content_type>"
  const parsed = new URL(uri);
  const id = parsed.host;
  const contentType = parsed.searchParams.get('content_type');
  const traceId = parsed.searchParams.get('trace_id');

  if (!id || !contentType || !traceId) {
    return null;
  }

  return {
    id,
    contentType,
    traceId,
  };
}

export const ModelTraceExplorerFieldRenderer = ({
  title,
  data,
  renderMode,
}: {
  title: string;
  data: string;
  renderMode: 'default' | 'json' | 'text';
}) => {
  const parsedData = useMemo(() => {
    try {
      return JSON.parse(data);
    } catch (e) {
      return data;
    }
  }, [data]);

  const dataIsString = isString(parsedData);
  const chatMessages = normalizeConversation(parsedData);
  const isChatTools = Array.isArray(parsedData) && parsedData.length > 0 && every(parsedData, isModelTraceChatTool);
  const isRetrieverDocuments =
    Array.isArray(parsedData) && parsedData.length > 0 && every(parsedData, isRetrieverDocument);

  if (renderMode === 'json') {
    return <ModelTraceExplorerCodeSnippet title={title} data={data} initialRenderMode={CodeSnippetRenderMode.JSON} />;
  }

  if (renderMode === 'text') {
    return <ModelTraceExplorerCodeSnippet title={title} data={data} initialRenderMode={CodeSnippetRenderMode.TEXT} />;
  }

  if (dataIsString) {
    const parsed = parseAttachmentUri(parsedData);
    if (parsed) {
      const { id, contentType, traceId } = parsed;
      return <ModelTraceExplorerAttachmentRenderer title={title} id={id} contentType={contentType} traceId={traceId} />;
    }

    return <ModelTraceExplorerTextFieldRenderer title={title} value={parsedData} />;
  }

  if (chatMessages && chatMessages.length > 0) {
    return <ModelTraceExplorerConversation messages={chatMessages} />;
  }

  if (isChatTools) {
    return <ModelTraceExplorerChatToolsRenderer title={title} tools={parsedData} />;
  }

  if (isRetrieverDocuments) {
    return <ModelTraceExplorerRetrieverFieldRenderer title={title} documents={parsedData} />;
  }

  return <ModelTraceExplorerCodeSnippet title={title} data={data} />;
};
