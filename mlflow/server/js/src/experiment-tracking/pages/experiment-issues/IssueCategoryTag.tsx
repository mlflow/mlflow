import { Tag, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
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

export const IssueCategoryTag = ({ category, componentId, insideClickableRow = false }: IssueCategoryTagProps) => {
  const { theme } = useDesignSystemTheme();
  const description = categoryDescription(category);
  const chip = (
    <Tag
      componentId={componentId}
      color={categoryColor(category)}
      // Tag's default role="status" is an aria-live region; a list of fifty would announce itself.
      role="presentation"
      css={{ ...TAG_MARGIN_RESET, pointerEvents: 'none' }}
    >
      {categoryLabel(category)}
    </Tag>
  );

  if (!description) return chip;
  return (
    <Tooltip componentId={`${componentId}.definition_tooltip`} content={description}>
      {/* Keep the row as one focus target; detail chips also expose descriptions on keyboard focus. */}
      <span
        data-testid="issue-category-tag"
        tabIndex={insideClickableRow ? undefined : 0}
        css={{
          display: 'inline-flex',
          pointerEvents: 'auto',
          borderRadius: theme.borders.borderRadiusSm,
          '&:focus-visible': {
            outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
            outlineOffset: 2,
          },
        }}
      >
        {chip}
      </span>
    </Tooltip>
  );
};
