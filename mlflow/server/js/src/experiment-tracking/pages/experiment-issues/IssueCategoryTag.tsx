import { useIntl } from 'react-intl';
import { Button, InfoSmallIcon, Popover, Tag, Tooltip } from '@databricks/design-system';
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
  const intl = useIntl();
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
  if (!insideClickableRow) {
    return (
      <span css={{ display: 'inline-flex', alignItems: 'center' }}>
        {chip}
        <Popover.Root componentId={`${componentId}.definition`}>
          <Popover.Trigger asChild>
            <Button
              componentId={`${componentId}.help`}
              size="small"
              type="tertiary"
              icon={<InfoSmallIcon />}
              aria-label={intl.formatMessage(
                {
                  defaultMessage: 'About category {category}',
                  description: 'Experiment Issues category explanation button',
                },
                { category },
              )}
            />
          </Popover.Trigger>
          <Popover.Content css={{ maxWidth: 320 }}>{description}</Popover.Content>
        </Popover.Root>
      </span>
    );
  }
  return (
    <Tooltip componentId={`${componentId}.definition_tooltip`} content={description}>
      {/* Unfocusable on purpose: it takes the pointer, the row still takes the click. */}
      <span data-testid="issue-category-tag" css={{ display: 'inline-flex', pointerEvents: 'auto' }}>
        {chip}
      </span>
    </Tooltip>
  );
};
