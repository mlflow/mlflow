import { useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { Assessment } from '../ModelTrace.types';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import {
  ModelTraceExplorerFieldRenderer,
  DEFAULT_MAX_VISIBLE_CHAT_MESSAGES,
} from '../field-renderers/ModelTraceExplorerFieldRenderer';

const DEFAULT_MAX_VISIBLE_ITEMS = 3;

export const ModelTraceExplorerSummarySection = ({
  title,
  data,
  renderMode,
  sectionKey,
  maxVisibleItems = DEFAULT_MAX_VISIBLE_ITEMS,
  maxVisibleChatMessages = DEFAULT_MAX_VISIBLE_CHAT_MESSAGES,
  className,
  chatMessageFormat,
  assessments,
}: {
  title: React.ReactElement;
  data: { key: string; value: string }[];
  renderMode: 'default' | 'json' | 'text';
  sectionKey: string;
  maxVisibleItems?: number;
  maxVisibleChatMessages?: number;
  className?: string;
  chatMessageFormat?: string;
  assessments?: Assessment[];
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  // In default mode, hide scalar fields when sibling fields contain attachment refs.
  // This gives a clean media-only view (e.g., DALL-E output shows images without "created" timestamp).
  const hasAttachmentRefs = data.some((item) => item.value.includes('mlflow-attachment://'));
  const filteredData =
    renderMode === 'default' && hasAttachmentRefs
      ? data.filter((item) => item.value.includes('mlflow-attachment://'))
      : data;
  const shouldTruncateItems = filteredData.length > maxVisibleItems;
  const visibleItems = shouldTruncateItems && !expanded ? filteredData.slice(-maxVisibleItems) : filteredData;
  const hiddenItemCount = shouldTruncateItems ? filteredData.length - visibleItems.length : 0;

  return (
    <ModelTraceExplorerCollapsibleSection withBorder title={title} className={className} sectionKey={sectionKey}>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {shouldTruncateItems && (
          <Typography.Link
            css={{ alignSelf: 'flex-start', marginLeft: theme.spacing.xs }}
            componentId="shared.model-trace-explorer.conversation-toggle"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Show less' : `Show ${hiddenItemCount} more`}
          </Typography.Link>
        )}
        {visibleItems.map(({ key, value }, index) => (
          <ModelTraceExplorerFieldRenderer
            key={key || index}
            title={key}
            data={value}
            renderMode={renderMode}
            chatMessageFormat={chatMessageFormat}
            maxVisibleMessages={maxVisibleChatMessages}
            assessments={assessments}
          />
        ))}
      </div>
    </ModelTraceExplorerCollapsibleSection>
  );
};
