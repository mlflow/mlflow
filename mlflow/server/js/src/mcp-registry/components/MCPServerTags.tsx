import { Overflow, Tag, useDesignSystemTheme } from '@databricks/design-system';

export const MCPServerTags = ({ tags }: { tags: Record<string, string> }) => {
  const { theme } = useDesignSystemTheme();
  const entries = Object.entries(tags);
  if (entries.length === 0) return <span aria-label="No tags">—</span>;
  return (
    <div
      css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm, maxWidth: '100%' }}
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
