import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import classnames from 'classnames';
import { forwardRef, useContext } from 'react';
import { TableContext } from './Table';
import { repeatingElementsStyles, tableClassNames } from './tableStyles';
import { Typography } from '../Typography';
export const TableCell = forwardRef(function ({ children, className, ellipsis = false, multiline = false, align = 'left', style, wrapContent = true, ...rest }, ref) {
    const { size, grid } = useContext(TableContext);
    let typographySize = 'md';
    if (size === 'small') {
        typographySize = 'sm';
    }
    const content = wrapContent === true ? (_jsx(Typography.Text, { ellipsis: !multiline, size: typographySize, title: (!multiline && typeof children === 'string' && children) || undefined, 
        // Needed for the button focus outline to be visible for the expand/collapse buttons
        css: { '&:has(> button)': { overflow: 'visible' } }, children: children })) : (children);
    return (_jsx("div", { ...rest, role: "cell", style: { textAlign: align, ...style }, ref: ref, 
        // PE-259 Use more performance className for grid but keep css= for compatibility.
        css: !grid ? repeatingElementsStyles.cell : undefined, className: classnames(grid && tableClassNames.cell, className), children: content }));
});
//# sourceMappingURL=TableCell.js.map