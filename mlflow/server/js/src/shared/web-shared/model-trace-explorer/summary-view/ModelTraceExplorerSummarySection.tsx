import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { useState } from 'react';

const DEFAULT_MAX_VISIBLE_ITEMS = 3;

export const ModelTraceExplorerSummarySection = ({
  title,
  data,
  renderMode,
  sectionKey,
  maxVisibleItems = DEFAULT_MAX_VISIBLE_ITEMS,
  className,
  chatMessageFormat,
}: {
  title: React.ReactElement;
  data: { key: string; value: string }[];
  renderMode: 'default' | 'json' | 'text';
  sectionKey: string;
  maxVisibleItems?: number;
  className?: string;
  chatMessageFormat?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const shouldTruncateItems = data.length > maxVisibleItems;

  const visibleItems = shouldTruncateItems && !expanded ? data.slice(-maxVisibleItems) : data;
  const hiddenItemCount = shouldTruncateItems ? data.length - visibleItems.length : 0;

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
          />
        ))}
      </div>
    </ModelTraceExplorerCollapsibleSection>
  );
};
