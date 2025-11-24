import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { forwardRef } from 'react';
import { useDesignSystemTheme } from '../Hooks';
export const TableFilterLayout = forwardRef(function TableFilterLayout({ children, style, className, actions, ...rest }, ref) {
    const { theme } = useDesignSystemTheme();
    const tableFilterLayoutStyles = {
        layout: css({
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            marginBottom: 'var(--table-filter-layout-group-margin)',
            columnGap: 'var(--table-filter-layout-group-margin)',
            rowGap: 'var(--table-filter-layout-item-gap)',
            flexWrap: 'wrap',
        }),
        filters: css({
            display: 'flex',
            flexWrap: 'wrap',
            flexDirection: 'row',
            alignItems: 'center',
            gap: 'var(--table-filter-layout-item-gap)',
            marginRight: 'var(--table-filter-layout-group-margin)',
            flex: 1,
        }),
        filterActions: css({
            display: 'flex',
            flexWrap: 'wrap',
            gap: 'var(--table-filter-layout-item-gap)',
            alignSelf: 'flex-start',
        }),
    };
    return (_jsxs("div", { ...rest, ref: ref, style: {
            ['--table-filter-layout-item-gap']: `${theme.spacing.sm}px`,
            ['--table-filter-layout-group-margin']: `${theme.spacing.md}px`,
            ...style,
        }, css: tableFilterLayoutStyles.layout, className: className, children: [_jsx("div", { css: tableFilterLayoutStyles.filters, children: children }), actions && _jsx("div", { css: tableFilterLayoutStyles.filterActions, children: actions })] }));
});
//# sourceMappingURL=TableFilterLayout.js.map