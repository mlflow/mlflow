import {
  Button,
  CheckIcon,
  ChevronDownIcon,
  DropdownMenu,
  SortAscendingIcon,
  SortDescendingIcon,
  Typography,
  useDesignSystemTheme,
  VisibleOffIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { SortDirection } from './types';

// Static componentId prefix — the `@databricks/no-dynamic-property-value` rule needs static literals.
const COMPONENT_ID = 'web-shared.traces-table.header-menu';

// Keep row/resize handlers from firing when the user interacts with the menu trigger.
const stopPropagation = (event: React.MouseEvent) => event.stopPropagation();

interface TraceColumnHeaderMenuProps {
  sortable: boolean;
  sortDirection: SortDirection | 'none';
  onSortAscending: () => void;
  onSortDescending: () => void;
  /** Omit → no Hide item (and, on a non-sortable column, no menu at all). */
  onHide?: () => void;
  triggerLabel: string;
}

/** The per-column header dropdown: Sort ascending/descending (sortable columns) and Hide column. */
export const TraceColumnHeaderMenu = ({
  sortable,
  sortDirection,
  onSortAscending,
  onSortDescending,
  onHide,
  triggerLabel,
}: TraceColumnHeaderMenuProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const iconCss = { marginRight: theme.spacing.sm };

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId={`${COMPONENT_ID}.trigger`}
          size="small"
          icon={<ChevronDownIcon />}
          aria-label={triggerLabel}
          onClick={stopPropagation}
        />
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="start">
        {sortable && (
          <>
            <DropdownMenu.Item componentId={`${COMPONENT_ID}.sort-ascending`} onClick={onSortAscending}>
              <SortAscendingIcon css={iconCss} />
              <FormattedMessage
                defaultMessage="Sort ascending"
                description="Traces table column header menu item to sort the column in ascending order"
              />
              {sortDirection === 'asc' && (
                <DropdownMenu.HintColumn>
                  <CheckIcon />
                </DropdownMenu.HintColumn>
              )}
            </DropdownMenu.Item>
            <DropdownMenu.Item componentId={`${COMPONENT_ID}.sort-descending`} onClick={onSortDescending}>
              <SortDescendingIcon css={iconCss} />
              <FormattedMessage
                defaultMessage="Sort descending"
                description="Traces table column header menu item to sort the column in descending order"
              />
              {sortDirection === 'desc' && (
                <DropdownMenu.HintColumn>
                  <CheckIcon />
                </DropdownMenu.HintColumn>
              )}
            </DropdownMenu.Item>
            {onHide && <DropdownMenu.Separator />}
          </>
        )}
        {onHide && (
          <DropdownMenu.Item componentId={`${COMPONENT_ID}.hide-column`} onClick={onHide}>
            <VisibleOffIcon css={iconCss} />
            <FormattedMessage
              defaultMessage="Hide column"
              description="Traces table column header menu item to hide the column"
            />
          </DropdownMenu.Item>
        )}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

export interface TraceColumnHeaderProps {
  label: React.ReactNode;
  /** Plain-text column name for the menu trigger's accessible label. */
  labelText?: string;
  sortable: boolean;
  sortDirection: SortDirection | 'none';
  onSortAscending: () => void;
  onSortDescending: () => void;
  onHide?: () => void;
}

/**
 * Header-cell content with a per-column options menu. Replaces DuBois `sortable` (can't nest a menu
 * inside its button trigger); sort moved into the menu dropdown.
 */
export const TraceColumnHeader = ({
  label,
  labelText,
  sortable,
  sortDirection,
  onSortAscending,
  onSortDescending,
  onHide,
}: TraceColumnHeaderProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const triggerLabel = labelText
    ? intl.formatMessage(
        {
          defaultMessage: 'Column options for {column}',
          description: 'Accessible label for the per-column options menu button in the traces table header',
        },
        { column: labelText },
      )
    : intl.formatMessage({
        defaultMessage: 'Column options',
        description: 'Accessible label for the per-column options menu button in the traces table header',
      });

  const showMenu = sortable || Boolean(onHide);

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, width: '100%', minWidth: 0 }}>
      <Typography.Text bold ellipsis className="table-header-text" css={{ minWidth: 0, flex: '0 1 auto' }}>
        {label}
      </Typography.Text>
      {sortable && sortDirection !== 'none' && (
        <span css={{ flexShrink: 0, display: 'inline-flex', color: theme.colors.textSecondary }}>
          {sortDirection === 'asc' ? <SortAscendingIcon /> : <SortDescendingIcon />}
        </span>
      )}
      {showMenu && (
        <span css={{ flexShrink: 0 }}>
          <TraceColumnHeaderMenu
            sortable={sortable}
            sortDirection={sortDirection}
            onSortAscending={onSortAscending}
            onSortDescending={onSortDescending}
            onHide={onHide}
            triggerLabel={triggerLabel}
          />
        </span>
      )}
    </div>
  );
};
