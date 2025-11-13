import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useContext } from 'react';
import { getOffsets, genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import { LoadingState } from '../LoadingState/LoadingState';
import { TableContext } from '../TableUI/Table';
import { TableCell } from '../TableUI/TableCell';
import { TableRow } from '../TableUI/TableRow';
import { TableRowAction } from '../TableUI/TableRowAction';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const TableSkeletonStyles = {
    container: css({
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
    }),
    cell: css({
        width: '100%',
        height: 8,
        borderRadius: 4,
        background: 'var(--table-skeleton-color)',
        marginTop: 'var(--table-skeleton-row-vertical-margin)',
        marginBottom: 'var(--table-skeleton-row-vertical-margin)',
    }),
};
export const TableSkeleton = ({ lines = 1, seed = '', frameRate = 60, style, label, ...rest }) => {
    const { theme } = useDesignSystemTheme();
    const { size } = useContext(TableContext);
    const widths = getOffsets(seed);
    return (_jsxs("div", { ...rest, ...addDebugOutlineIfEnabled(), "aria-busy": true, css: TableSkeletonStyles.container, role: "status", style: {
            ...style,
            // TODO: Pull this from the themes; it's not currently available.
            ['--table-skeleton-color']: theme.isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(31, 38, 45, 0.1)',
            ['--table-skeleton-row-vertical-margin']: size === 'small' ? '4px' : '6px',
        }, children: [[...Array(lines)].map((_, idx) => (_jsx("div", { css: [
                    TableSkeletonStyles.cell,
                    genSkeletonAnimatedColor(theme, frameRate),
                    { width: `calc(100% - ${widths[idx % widths.length]}px)` },
                ] }, idx))), _jsx("span", { css: visuallyHidden, children: label })] }));
};
export const TableSkeletonRows = ({ table, actionColumnIds = [], numRows = 3, loading = true, loadingDescription = 'Table skeleton rows', }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsxs(_Fragment, { children: [loading && _jsx(LoadingState, { description: loadingDescription }), [...Array(numRows).keys()].map((i) => (_jsx(TableRow, { children: table.getFlatHeaders().map((header) => {
                    const meta = header.column.columnDef.meta;
                    return actionColumnIds.includes(header.id) ? (_jsx(TableRowAction, { children: _jsx(TableSkeleton, { style: { width: theme.general.iconSize } }) }, `cell-${header.id}-${i}`)) : (_jsx(TableCell, { style: meta?.styles ?? (meta?.width !== undefined ? { maxWidth: meta.width } : {}), children: _jsx(TableSkeleton, { seed: `skeleton-${header.id}-${i}`, lines: meta?.numSkeletonLines ?? undefined }) }, `cell-${header.id}-${i}`));
                }) }, i)))] }));
};
//# sourceMappingURL=TableSkeleton.js.map