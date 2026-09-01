import { Fragment } from 'react';
import { Button, ColumnsIcon, DropdownMenu, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { TraceColumnId } from './types';

// Module-local static prefix: the `@databricks/no-dynamic-property-value` lint rule requires every
// `componentId` to be statically determinable, so an in-file const (resolved to a literal) is the
// only way to share a namespace across this file's ids. An imported const is not resolved by the rule.
const COMPONENT_ID = 'web-shared.traces-table';

/**
 * One selectable column in the dropdown. `label` is already localized. Each option carries its own
 * static `componentId` (destructured in the render map) so per-column analytics ids stay distinct
 * while satisfying the static-componentId lint rule.
 */
export interface ColumnSelectorOption {
  id: TraceColumnId;
  label: React.ReactNode;
  componentId: string;
  /** Renders the item checked-but-unclickable (e.g. a column the current view forces on). */
  disabled?: boolean;
}

/** A dynamic-id selectable column (e.g. a product-specific column), for use inside a group. */
export interface GenericColumnOption {
  id: string;
  label: React.ReactNode;
  componentId: string;
}

/**
 * A labeled section of dynamic-id columns rendered below the standard columns. Generic on purpose:
 * the consumer supplies the `label`, so the selector stays free of any product vocabulary.
 */
export interface ColumnSelectorGroup {
  label: React.ReactNode;
  options: GenericColumnOption[];
  visibleIds: string[];
  onToggle: (id: string) => void;
}

export interface TraceColumnSelectorProps {
  /** The columns offered, in dropdown order. */
  columns: ColumnSelectorOption[];
  visibleColumns: TraceColumnId[];
  onToggleColumn: (column: TraceColumnId) => void;
  onResetToDefaults: () => void;
  /** Optional labeled sections of dynamic-id columns, each rendered under its own header. */
  groups?: ColumnSelectorGroup[];
}

/**
 * Generic column-visibility dropdown, driven entirely by the `columns` option list. A checkbox per
 * column reflects/toggles its visibility, plus a "Reset to defaults" item. The trigger shows the
 * visible/total count.
 */
export const TraceColumnSelector: React.FC<TraceColumnSelectorProps> = ({
  columns,
  visibleColumns,
  onToggleColumn,
  onResetToDefaults,
  groups,
}: TraceColumnSelectorProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const visible = new Set(visibleColumns);

  return (
    // Non-modal so the open menu doesn't aria-hide / scroll-lock the table behind it — the grid stays
    // visible and interactive while columns are toggled (and it keeps the menu from trapping focus).
    <DropdownMenu.Root modal={false}>
      <Tooltip
        componentId={`${COMPONENT_ID}.column-selector.trigger.tooltip`}
        content={intl.formatMessage(
          {
            defaultMessage: 'Columns ({visible}/{total})',
            description: 'Column-selector trigger tooltip showing visible columns out of total available',
          },
          { visible: visibleColumns.length, total: columns.length },
        )}
      >
        <DropdownMenu.Trigger asChild>
          <Button
            componentId={`${COMPONENT_ID}.column-selector.trigger`}
            icon={<ColumnsIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Select visible columns',
              description: 'Aria label for the column-selector dropdown trigger on the traces table',
            })}
            // Icon-only DS Buttons are intentionally borderless: Button runs its `border: none` through
            // importantify AND puts it on the `&.<prefix>-btn-icon-only` selector (0,2,0). Beating that needs
            // BOTH `!important` (to tie its importance) AND higher specificity — `&&&` (0,3,0) wins the tie.
            // Restores the toolbar border (matching the search box) via a theme token, so it adapts to light + dark.
            css={{ '&&&': { border: `1px solid ${theme.colors.actionDefaultBorderDefault} !important` } }}
          />
        </DropdownMenu.Trigger>
      </Tooltip>
      <DropdownMenu.Content align="end">
        {columns.map(({ id, label, componentId }) => (
          <DropdownMenu.CheckboxItem
            key={id}
            componentId={componentId}
            checked={visible.has(id)}
            // Toggle in onSelect (not onCheckedChange) and preventDefault so the menu stays open across
            // changes — several columns can be toggled in one visit. preventDefault here also suppresses
            // onCheckedChange in this Radix version, so the explicit onToggleColumn call is what fires.
            onSelect={(event) => {
              event.preventDefault();
              onToggleColumn(id);
            }}
          >
            <DropdownMenu.ItemIndicator />
            {label}
          </DropdownMenu.CheckboxItem>
        ))}
        {groups?.map((group, groupIndex) => {
          const groupVisible = new Set(group.visibleIds);
          return (
            <Fragment key={groupIndex}>
              <DropdownMenu.Separator />
              <DropdownMenu.Label>{group.label}</DropdownMenu.Label>
              {group.options.map(({ id, label, componentId }) => (
                <DropdownMenu.CheckboxItem
                  key={id}
                  componentId={componentId}
                  checked={groupVisible.has(id)}
                  onSelect={(event) => {
                    event.preventDefault();
                    group.onToggle(id);
                  }}
                >
                  <DropdownMenu.ItemIndicator />
                  {label}
                </DropdownMenu.CheckboxItem>
              ))}
            </Fragment>
          );
        })}
        <DropdownMenu.Separator />
        <DropdownMenu.Item componentId={`${COMPONENT_ID}.column-selector.reset`} onClick={onResetToDefaults}>
          <FormattedMessage
            defaultMessage="Reset to defaults"
            description="Menu item that resets the traces table column visibility to defaults"
          />
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
