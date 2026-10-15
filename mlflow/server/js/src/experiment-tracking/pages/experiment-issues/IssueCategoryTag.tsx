import { Tag, Tooltip } from '@databricks/design-system';
import { categoryColor, categoryLabel, getCategoryDefinition } from './issueLabels';

export const IssueCategoryTag = ({
  category,
  componentId,
  insideClickableRow = false,
}: {
  category: string;
  componentId: string;
  insideClickableRow?: boolean;
}) => {
  const definition = getCategoryDefinition(category);
  const tag = (
    <Tag
      componentId={componentId}
      color={categoryColor(category)}
      role="presentation"
      css={{ marginRight: 0, ...(insideClickableRow && { pointerEvents: 'none' }) }}
    >
      {categoryLabel(category)}
    </Tag>
  );
  if (!definition) return tag;
  return (
    <Tooltip componentId={`${componentId}.definition`} content={definition.description}>
      {/* Keep category hover help without moving focus from the clickable row to Tag's wrapper. */}
      <span css={{ display: 'inline-flex', pointerEvents: 'auto' }}>{tag}</span>
    </Tooltip>
  );
};
