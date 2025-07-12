import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import classnames from 'classnames';
import { createContext, forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import tableStyles from './tableStyles';
import { DesignSystemEventSuppressInteractionProviderContext, DesignSystemEventSuppressInteractionTrueContextValue, } from '../DesignSystemEventProvider/DesignSystemEventSuppressInteractionProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const TableContext = createContext({
    size: 'default',
    grid: false,
});
export const Table = forwardRef(function Table({ children, size = 'default', someRowsSelected, style, pagination, empty, className, scrollable = false, grid = false, noMinHeight = false, ...rest }, ref) {
    const { theme } = useDesignSystemTheme();
    const { useNewBorderColors } = useDesignSystemSafexFlags();
    const tableContentRef = useRef(null);
    useImperativeHandle(ref, () => tableContentRef.current);
    const minHeightCss = noMinHeight ? {} : { minHeight: !empty && pagination ? 150 : 100 };
    return (_jsx(DesignSystemEventSuppressInteractionProviderContext.Provider, { value: DesignSystemEventSuppressInteractionTrueContextValue, children: _jsx(TableContext.Provider, { value: useMemo(() => {
                return { size, someRowsSelected, grid };
            }, [size, someRowsSelected, grid]), children: _jsxs("div", { ...addDebugOutlineIfEnabled(), ...rest, 
                // This is a performance optimization; we want to statically create the styles for the table,
                // but for the dynamic theme values, we need to use CSS variables.
                // See: https://emotion.sh/docs/best-practices#advanced-css-variables-with-style
                style: {
                    ...style,
                    ['--table-header-active-color']: theme.colors.actionDefaultTextPress,
                    ['colorScheme']: theme.isDarkMode ? 'dark' : undefined,
                    ['--table-header-background-color']: theme.colors.backgroundPrimary,
                    ['--table-header-focus-color']: theme.colors.actionDefaultTextHover,
                    ['--table-header-sort-icon-color']: theme.colors.textSecondary,
                    ['--table-header-text-color']: theme.colors.actionDefaultTextDefault,
                    ['--table-row-hover']: theme.colors.tableRowHover,
                    ['--table-separator-color']: useNewBorderColors
                        ? theme.colors.border
                        : theme.colors.borderDecorative,
                    ['--table-resize-handle-color']: theme.colors.borderDecorative,
                    ['--table-spacing-md']: `${theme.spacing.md}px`,
                    ['--table-spacing-sm']: `${theme.spacing.sm}px`,
                    ['--table-spacing-xs']: `${theme.spacing.xs}px`,
                }, css: [tableStyles.tableWrapper, minHeightCss], className: classnames({
                    'table-isScrollable': scrollable,
                    'table-isGrid': grid,
                }, className), children: [_jsxs("div", { role: "table", ref: tableContentRef, css: tableStyles.table, 
                        // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
                        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
                        tabIndex: scrollable ? 0 : -1, children: [children, empty && _jsx("div", { css: { padding: theme.spacing.lg }, children: empty })] }), !empty && pagination && _jsx("div", { css: tableStyles.paginationContainer, children: pagination })] }) }) }));
});
//# sourceMappingURL=Table.js.map