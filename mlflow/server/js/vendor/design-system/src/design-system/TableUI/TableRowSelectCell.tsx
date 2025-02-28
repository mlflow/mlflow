import { forwardRef, useContext } from 'react';

import { TableContext } from './Table';
import { TableRowContext } from './TableRow';
import tableStyles from './tableStyles';
import { Checkbox } from '../Checkbox';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { AnalyticsEventProps, HTMLDataAttributes } from '../types';

interface TableRowSelectCellProps
  extends HTMLDataAttributes,
    React.HTMLAttributes<HTMLDivElement>,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  /** Called when the checkbox is clicked */
  onChange?: (event: unknown) => void;
  /** Whether the checkbox is checked */
  checked?: boolean;
  /** Whether the row is indeterminate. Should only be used in header rows. */
  indeterminate?: boolean;
  /** Don't render a checkbox; used for providing spacing in header if you don't want "Select All" functionality */
  noCheckbox?: boolean;
  /** Whether the checkbox is disabled */
  isDisabled?: boolean;
  checkboxLabel?: string;
}

export const TableRowSelectCell = forwardRef<HTMLDivElement, TableRowSelectCellProps>(function TableRowSelectCell(
  {
    onChange,
    checked,
    indeterminate,
    noCheckbox,
    children,
    isDisabled,
    checkboxLabel,
    componentId,
    analyticsEvents,
    ...rest
  },
  ref,
) {
  const { theme } = useDesignSystemTheme();
  const { isHeader } = useContext(TableRowContext);
  const { someRowsSelected } = useContext(TableContext);

  if (typeof someRowsSelected === 'undefined') {
    throw new Error(
      '`TableRowSelectCell` cannot be used unless `someRowsSelected` has been provided to the `Table` component, see documentation.',
    );
  }

  if (!isHeader && indeterminate) {
    throw new Error('`TableRowSelectCell` cannot be used with `indeterminate` in a non-header row.');
  }

  return (
    <div
      {...rest}
      ref={ref}
      css={tableStyles.checkboxCell}
      style={{
        ['--row-checkbox-opacity' as any]: someRowsSelected ? 1 : 0,
        zIndex: theme.options.zIndexBase,
      }}
      role={isHeader ? 'columnheader' : 'cell'}
      // TODO: Ideally we shouldn't need to specify this `className`, but it allows for row-hovering to reveal
      // the checkbox in `TableRow`'s CSS without extra JS pointerin/out events.
      className="table-row-select-cell"
    >
      {!noCheckbox && (
        <Checkbox
          componentId={componentId}
          analyticsEvents={analyticsEvents}
          isChecked={checked || (indeterminate && null)}
          onChange={(_checked, event) => onChange?.(event.nativeEvent)}
          isDisabled={isDisabled}
          aria-label={checkboxLabel}
        />
      )}
    </div>
  );
});
