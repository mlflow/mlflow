import { Fragment } from 'react';
import { Button, ChevronDownIcon, ColumnsIcon, DropdownMenu } from '@databricks/design-system';
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
  const visible = new Set(visibleColumns);

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId={`${COMPONENT_ID}.column-selector.trigger`}
          icon={<ColumnsIcon />}
          endIcon={<ChevronDownIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Select visible columns',
            description: 'Aria label for the column-selector dropdown trigger on the traces table',
          })}
        >
          <FormattedMessage
            defaultMessage="Columns ({visible}/{total})"
            description="Column-selector trigger label showing visible columns out of total available"
            values={{ visible: visibleColumns.length, total: columns.length }}
          />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        {columns.map(({ id, label, componentId }) => (
          <DropdownMenu.CheckboxItem
            key={id}
            componentId={componentId}
            checked={visible.has(id)}
            onCheckedChange={() => onToggleColumn(id)}
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
                  onCheckedChange={() => group.onToggle(id)}
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
