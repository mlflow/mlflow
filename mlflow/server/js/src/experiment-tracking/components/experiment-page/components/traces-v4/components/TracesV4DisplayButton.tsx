import { Fragment } from 'react';
import {
  Button,
  ChevronDownIcon,
  ColumnsIcon,
  DropdownMenu,
  RowsIcon,
  SlidersIcon,
  SortAscendingIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import {
  ReorderableTraceColumnList,
  SORTABLE_TRACE_COLUMNS,
  ToolbarCollapsibleLabel,
  type ColumnSelectorGroup,
  type ReorderableTraceColumnOption,
  type SortDirection,
  type TraceColumnId,
} from '@databricks/web-shared/traces-table';
import type { TracesV4Density } from '../hooks/useTracesV4Density';

/**
 * A labeled column group (e.g. assessments) whose header doubles as a show/hide-all toggle when
 * `onToggleAll` is supplied — otherwise the header is a plain, non-interactive label.
 */
export interface TracesV4ColumnGroup extends ColumnSelectorGroup {
  onToggleAll?: (visible: boolean) => void;
  /**
   * Render only the group's toggle-all header, not its per-column checkboxes. Used when the group's
   * columns already appear elsewhere (e.g. assessments live in the reorderable list) and the group
   * exists solely to add the bulk show/hide-all affordance, avoiding duplicate checkboxes.
   */
  headerOnly?: boolean;
}

// Module-local static analytics-id namespace (the `@databricks/no-dynamic-property-value` lint rule
// requires statically-determinable componentId values, so a runtime-injected prefix isn't possible).
const COMPONENT_ID = 'mlflow.traces-v4.display';

// A `RadioGroup` value must be a string; direction is joined onto the column to form one radio value.
const SORT_VALUE_SEPARATOR = ':';

export interface TracesV4DisplayButtonProps {
  onResetColumns: () => void;
  /** Human-readable label per sortable column id, for the Sort submenu options + the trigger hint. */
  sortColumnLabels: Record<TraceColumnId, React.ReactNode>;
  sort: TraceColumnId;
  dir: SortDirection;
  onSort: (column: TraceColumnId, direction: SortDirection) => void;
  density: TracesV4Density;
  onDensityChange: (density: TracesV4Density) => void;
  /** Column reordering (drag or Ctrl+Arrow): reorderable columns + visibility + handlers. */
  reorder: {
    columns: ReorderableTraceColumnOption[];
    visibleColumns: string[];
    onToggleColumn: (id: string) => void;
    onReorderColumn: (activeColumn: string, targetColumn: string) => void;
  };
  /**
   * Optional labeled sections of dynamic-id columns (tags / metadata) rendered as checkbox groups
   * beneath the reorderable list. These columns are toggle-only (not yet drag-reorderable).
   */
  columnGroups?: TracesV4ColumnGroup[];
}

/**
 * Consolidated "Display" popover for the V4 traces toolbar: column visibility, sort, and row height,
 * replacing the standalone columns button. Columns toggle in place (the menu stays open across
 * changes); Sort and Row height are single-choice radio groups whose current value shows as a hint on
 * the submenu trigger. All state is owned by the consumer and passed in — this component only renders.
 */
export const TracesV4DisplayButton = ({
  onResetColumns,
  sortColumnLabels,
  sort,
  dir,
  onSort,
  density,
  onDensityChange,
  reorder,
  columnGroups,
}: TracesV4DisplayButtonProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const sortValue = `${sort}${SORT_VALUE_SEPARATOR}${dir}`;
  const rowHeightLabel =
    density === 'small'
      ? intl.formatMessage({
          defaultMessage: 'Compact',
          description: 'Traces table row-height option: compact (dense) rows',
        })
      : density === 'standard'
        ? intl.formatMessage({
            defaultMessage: 'Standard',
            description: 'Traces table row-height option: taller rows with two-line text previews',
          })
        : intl.formatMessage({
            defaultMessage: 'Tall',
            description: 'Traces table row-height option: tall rows with longer text previews',
          });

  return (
    // Non-modal so the open menu doesn't aria-hide / scroll-lock the table behind it — the grid stays
    // visible and interactive while display options are changed.
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId={`${COMPONENT_ID}.trigger`}
          icon={<SlidersIcon css={{ color: theme.colors.textSecondary }} />}
          endIcon={<ChevronDownIcon />}
          // Names the button when its label collapses to icon-only.
          aria-label={intl.formatMessage({
            defaultMessage: 'Display',
            description: 'Label for the Display (columns / sort / row height) toolbar button on the V4 traces tab',
          })}
        >
          <ToolbarCollapsibleLabel>
            <FormattedMessage
              defaultMessage="Display"
              description="Label for the Display (columns / sort / row height) toolbar button on the V4 traces tab"
            />
          </ToolbarCollapsibleLabel>
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        <DropdownMenu.Sub>
          <DropdownMenu.SubTrigger>
            <DropdownMenu.IconWrapper>
              <ColumnsIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Columns"
              description="Display menu item opening the traces table column-visibility submenu"
            />
          </DropdownMenu.SubTrigger>
          <DropdownMenu.SubContent>
            {/* One drag/keyboard-reorderable list built by the consumer from the merged standard +
                assessment columns (in persisted order); toggles are routed by id. This replaces the
                plain per-column checkboxes + assessment group so visibility and ordering share one UI. */}
            <ReorderableTraceColumnList {...reorder} />
            {/* Dynamic tag/metadata columns render as checkbox groups beneath the reorderable list.
                They are toggle-only for now (not yet part of the drag-reorderable order). */}
            {columnGroups?.map((group, groupIndex) => {
              const groupVisible = new Set(group.visibleIds);
              const allVisible = group.options.length > 0 && group.options.every(({ id }) => groupVisible.has(id));
              return (
                <Fragment key={groupIndex}>
                  <DropdownMenu.Separator />
                  {group.onToggleAll ? (
                    // Header doubles as a show/hide-all toggle: checked when every column in the group
                    // is visible, one click flips them all. Bold + primary weight reads as a section
                    // header rather than a regular item.
                    <DropdownMenu.CheckboxItem
                      componentId={`${COMPONENT_ID}.columns.group-toggle-all`}
                      checked={allVisible}
                      onSelect={(event) => {
                        event.preventDefault();
                        group.onToggleAll?.(!allVisible);
                      }}
                      css={{
                        fontWeight: theme.typography.typographyBoldFontWeight,
                        color: theme.colors.textPrimary,
                        marginTop: theme.spacing.xs,
                      }}
                    >
                      <DropdownMenu.ItemIndicator />
                      {group.label}
                    </DropdownMenu.CheckboxItem>
                  ) : (
                    <DropdownMenu.Label>{group.label}</DropdownMenu.Label>
                  )}
                  {/* `headerOnly` groups contribute just the toggle-all header — their columns are
                      already listed in the reorderable list above, so skip the per-item checkboxes. */}
                  {!group.headerOnly &&
                    group.options.map(({ id, label, componentId }) => (
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
            <DropdownMenu.Item componentId={`${COMPONENT_ID}.columns.reset`} onClick={onResetColumns}>
              <FormattedMessage
                defaultMessage="Reset to defaults"
                description="Menu item that resets the traces table column visibility to defaults"
              />
            </DropdownMenu.Item>
          </DropdownMenu.SubContent>
        </DropdownMenu.Sub>

        <DropdownMenu.Sub>
          <DropdownMenu.SubTrigger>
            <DropdownMenu.IconWrapper>
              <SortAscendingIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Sort"
              description="Display menu item opening the traces table sort submenu"
            />
            {/* Grow + right-align so the value sits flush against the chevron the SubTrigger appends;
                the default HintColumn `margin-left: auto` would otherwise fight the chevron's own
                auto-margin and float the value mid-row. */}
            <DropdownMenu.HintColumn css={{ flexGrow: 1, textAlign: 'right', marginLeft: theme.spacing.sm }}>
              {sortColumnLabels[sort]}
            </DropdownMenu.HintColumn>
          </DropdownMenu.SubTrigger>
          <DropdownMenu.SubContent>
            <DropdownMenu.RadioGroup
              componentId={`${COMPONENT_ID}.sort`}
              value={sortValue}
              onValueChange={(value) => {
                const [column, direction] = value.split(SORT_VALUE_SEPARATOR);
                onSort(column as TraceColumnId, direction as SortDirection);
              }}
            >
              {SORTABLE_TRACE_COLUMNS.map((column) => (
                <Fragment key={column}>
                  <DropdownMenu.RadioItem value={`${column}${SORT_VALUE_SEPARATOR}desc`}>
                    <DropdownMenu.ItemIndicator />
                    <FormattedMessage
                      defaultMessage="{column} (descending)"
                      description="Traces table sort option: sort by a column, newest/largest first"
                      values={{ column: sortColumnLabels[column] }}
                    />
                  </DropdownMenu.RadioItem>
                  <DropdownMenu.RadioItem value={`${column}${SORT_VALUE_SEPARATOR}asc`}>
                    <DropdownMenu.ItemIndicator />
                    <FormattedMessage
                      defaultMessage="{column} (ascending)"
                      description="Traces table sort option: sort by a column, oldest/smallest first"
                      values={{ column: sortColumnLabels[column] }}
                    />
                  </DropdownMenu.RadioItem>
                </Fragment>
              ))}
            </DropdownMenu.RadioGroup>
          </DropdownMenu.SubContent>
        </DropdownMenu.Sub>

        <DropdownMenu.Sub>
          <DropdownMenu.SubTrigger>
            <DropdownMenu.IconWrapper>
              <RowsIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Row height"
              description="Display menu item opening the traces table row-height submenu"
            />
            <DropdownMenu.HintColumn css={{ flexGrow: 1, textAlign: 'right', marginLeft: theme.spacing.sm }}>
              {rowHeightLabel}
            </DropdownMenu.HintColumn>
          </DropdownMenu.SubTrigger>
          <DropdownMenu.SubContent>
            <DropdownMenu.RadioGroup
              componentId={`${COMPONENT_ID}.row-height`}
              value={density}
              onValueChange={(value) => onDensityChange(value as TracesV4Density)}
            >
              <DropdownMenu.RadioItem value="small">
                <DropdownMenu.ItemIndicator />
                <FormattedMessage
                  defaultMessage="Compact"
                  description="Traces table row-height option: compact (dense) rows"
                />
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem value="standard">
                <DropdownMenu.ItemIndicator />
                <FormattedMessage
                  defaultMessage="Standard"
                  description="Traces table row-height option: taller rows with two-line text previews"
                />
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem value="tall">
                <DropdownMenu.ItemIndicator />
                <FormattedMessage
                  defaultMessage="Tall"
                  description="Traces table row-height option: tall rows with longer text previews"
                />
              </DropdownMenu.RadioItem>
            </DropdownMenu.RadioGroup>
          </DropdownMenu.SubContent>
        </DropdownMenu.Sub>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
