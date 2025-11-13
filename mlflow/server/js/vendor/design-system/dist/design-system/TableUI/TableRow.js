import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import classnames from 'classnames';
import { createContext, forwardRef, useContext, useMemo } from 'react';
import { TableContext } from './Table';
import { hideIconButtonRowStyles, repeatingElementsStyles, tableClassNames } from './tableStyles';
import { useDesignSystemTheme } from '../Hooks';
export const TableRowContext = createContext({
    isHeader: false,
});
export const TableRow = forwardRef(function TableRow({ children, className, style, isHeader = false, skipIconHiding = false, verticalAlignment, ...rest }, ref) {
    const { size, grid } = useContext(TableContext);
    const { theme } = useDesignSystemTheme();
    // Vertical only be larger if the row is a header AND size is large.
    const shouldUseLargeVerticalPadding = isHeader && size === 'default';
    let rowPadding;
    if (shouldUseLargeVerticalPadding) {
        rowPadding = theme.spacing.sm;
    }
    else if (size === 'default') {
        rowPadding = 6;
    }
    else {
        rowPadding = theme.spacing.xs;
    }
    return (_jsx(TableRowContext.Provider, { value: useMemo(() => {
            return { isHeader };
        }, [isHeader]), children: _jsx("div", { ...rest, ref: ref, role: "row", style: {
                ...style,
                ['--table-row-vertical-padding']: `${rowPadding}px`,
            }, 
            // PE-259 Use more performance className for grid but keep css= for consistency.
            css: [!isHeader && !skipIconHiding && hideIconButtonRowStyles, !grid && repeatingElementsStyles.row], className: classnames(className, grid && tableClassNames.row, {
                'table-isHeader': isHeader,
                'table-row-isGrid': grid,
            }), children: children }) }));
});
//# sourceMappingURL=TableRow.js.map