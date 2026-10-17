import { Tag, Tooltip } from '@databricks/design-system';
import { categoryColor, categoryDescription, categoryLabel, TAG_MARGIN_RESET } from './issueLabels';

export interface IssueCategoryTagProps {
  category: string;
  /** Prefix for the chip's own analytics id; the tooltip's is derived from it. */
  componentId: string;
  /**
   * Inside a `role="button"` row, a click reaching `Tag`'s `tabIndex={-1}` wrapper would take focus
   * off the row. Set here, only the tooltip trigger takes pointer events, so the click reaches it.
   */
  insideClickableRow?: boolean;
}

/**
 * A Tooltip, not a Popover: the list row is one `role="button"`, so an interactive control nested
 * inside it is an accessibility bug. Cost: the definition is hover-only.
 */
export const IssueCategoryTag = ({ category, componentId, insideClickableRow = false }: IssueCategoryTagProps) => {
  const description = categoryDescription(category);
  const chip = (
    <Tag
      componentId={componentId}
      color={categoryColor(category)}
      // Tag's default role="status" is an aria-live region; a list of fifty would announce itself.
      role="presentation"
      css={{ ...TAG_MARGIN_RESET, ...(insideClickableRow && { pointerEvents: 'none' }) }}
    >
      {categoryLabel(category)}
    </Tag>
  );

  if (!description) return chip;
  return (
    <Tooltip componentId={`${componentId}.definition_tooltip`} content={description}>
      {/* Unfocusable on purpose: it takes the pointer, the row still takes the click. */}
      <span data-testid="issue-category-tag" css={{ display: 'inline-flex', pointerEvents: 'auto' }}>
        {chip}
      </span>
    </Tooltip>
  );
};
