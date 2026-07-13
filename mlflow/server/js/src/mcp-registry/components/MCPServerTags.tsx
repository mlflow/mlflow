import { Overflow, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { inlineFlexRowStyles } from '../styles';

export const MCPServerTags = ({ tags }: { tags: Record<string, string> }) => {
  const { theme } = useDesignSystemTheme();
  const entries = Object.entries(tags);
  if (entries.length === 0) return <span aria-label="No tags">—</span>;
  return (
    <div
      css={inlineFlexRowStyles(theme)}
      onClick={(e) => {
        if ((e.target as HTMLElement).closest('button')) {
          e.stopPropagation();
        }
      }}
    >
      <Overflow noMargin>
        {entries.map(([key, value]) => (
          <Tag key={key} componentId="mlflow.mcp_registry.tag">
            {value ? `${key}: ${value}` : key}
          </Tag>
        ))}
      </Overflow>
    </div>
  );
};
