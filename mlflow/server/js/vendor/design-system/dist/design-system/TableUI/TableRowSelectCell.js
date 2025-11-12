import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef, useContext } from 'react';
import { TableContext } from './Table';
import { TableRowContext } from './TableRow';
import tableStyles from './tableStyles';
import { Checkbox } from '../Checkbox';
import { useDesignSystemTheme } from '../Hooks';
export const TableRowSelectCell = forwardRef(function TableRowSelectCell({ onChange, checked, indeterminate, noCheckbox, children, isDisabled, checkboxLabel, componentId, analyticsEvents, ...rest }, ref) {
    const { theme } = useDesignSystemTheme();
    const { isHeader } = useContext(TableRowContext);
    const { someRowsSelected } = useContext(TableContext);
    if (typeof someRowsSelected === 'undefined') {
        throw new Error('`TableRowSelectCell` cannot be used unless `someRowsSelected` has been provided to the `Table` component, see documentation.');
    }
    if (!isHeader && indeterminate) {
        throw new Error('`TableRowSelectCell` cannot be used with `indeterminate` in a non-header row.');
    }
    return (_jsx("div", { ...rest, ref: ref, css: tableStyles.checkboxCell, style: {
            ['--row-checkbox-opacity']: someRowsSelected ? 1 : 0,
            zIndex: theme.options.zIndexBase,
        }, role: isHeader ? 'columnheader' : 'cell', 
        // TODO: Ideally we shouldn't need to specify this `className`, but it allows for row-hovering to reveal
        // the checkbox in `TableRow`'s CSS without extra JS pointerin/out events.
        className: "table-row-select-cell", children: !noCheckbox && (_jsx(Checkbox, { componentId: componentId, analyticsEvents: analyticsEvents, isChecked: checked || (indeterminate && null), onChange: (_checked, event) => onChange?.(event.nativeEvent), isDisabled: isDisabled, "aria-label": checkboxLabel })) }));
});
//# sourceMappingURL=TableRowSelectCell.js.map