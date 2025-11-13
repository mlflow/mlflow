import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import classnames from 'classnames';
import { forwardRef, useContext } from 'react';
import { TableContext } from './Table';
import { TableRowContext } from './TableRow';
import { hideIconButtonActionCellClassName } from './tableStyles';
import { useDesignSystemTheme } from '../Hooks';
const TableRowActionStyles = {
    container: css({
        width: 32,
        paddingTop: 'var(--vertical-padding)',
        paddingBottom: 'var(--vertical-padding)',
        display: 'flex',
        alignItems: 'start',
        justifyContent: 'center',
    }),
};
export const TableRowAction = forwardRef(function TableRowAction({ children, style, className, ...rest }, ref) {
    const { size } = useContext(TableContext);
    const { isHeader } = useContext(TableRowContext);
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { ...rest, ref: ref, role: isHeader ? 'columnheader' : 'cell', style: {
            ...style,
            ['--vertical-padding']: size === 'default' ? `${theme.spacing.xs}px` : 0,
        }, css: TableRowActionStyles.container, className: classnames(className, !isHeader && hideIconButtonActionCellClassName), children: children }));
});
/** @deprecated Use `TableRowAction` instead */
export const TableRowMenuContainer = TableRowAction;
//# sourceMappingURL=TableRowAction.js.map