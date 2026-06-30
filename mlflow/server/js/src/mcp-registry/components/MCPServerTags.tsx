import { Overflow, Tag } from '@databricks/design-system';

export const MCPServerTags = ({ tags }: { tags: Record<string, string> }) => {
  const entries = Object.entries(tags);
  if (entries.length === 0) return <span aria-label="No tags">—</span>;
  return (
    <Overflow noMargin>
      {entries.map(([key, value]) => (
        <Tag key={key} componentId="mlflow.mcp_registry.tag">
          {value ? `${key}: ${value}` : key}
        </Tag>
      ))}
    </Overflow>
  );
};
