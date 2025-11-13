import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { forwardRef } from 'react';
import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks';
import { SearchIcon } from '../Icon';
import { Input } from '../Input';
const getTableFilterInputStyles = (theme, defaultWidth) => {
    return css({
        [theme.responsive.mediaQueries.sm]: {
            width: 'auto',
        },
        [theme.responsive.mediaQueries.lg]: {
            width: '30%',
        },
        [theme.responsive.mediaQueries.xxl]: {
            width: defaultWidth,
        },
    });
};
export const TableFilterInput = forwardRef(function SearchInput({ onSubmit, showSearchButton, className, containerProps, searchButtonProps, ...inputProps }, ref) {
    const { theme } = useDesignSystemTheme();
    const DEFAULT_WIDTH = 400;
    let component = _jsx(Input, { prefix: _jsx(SearchIcon, {}), allowClear: true, ...inputProps, className: className, ref: ref });
    if (showSearchButton) {
        component = (_jsxs(Input.Group, { css: {
                display: 'flex',
                width: '100%',
            }, className: className, children: [_jsx(Input, { allowClear: true, ...inputProps, ref: ref, css: {
                        flex: 1,
                    } }), _jsx(Button, { componentId: inputProps.componentId
                        ? `${inputProps.componentId}.search_submit`
                        : 'codegen_design-system_src_design-system_tableui_tablefilterinput.tsx_65', htmlType: "submit", "aria-label": "Search", ...searchButtonProps, children: _jsx(SearchIcon, {}) })] }));
    }
    return (_jsx("div", { style: {
            height: theme.general.heightSm,
        }, css: getTableFilterInputStyles(theme, DEFAULT_WIDTH), ...containerProps, children: onSubmit ? (_jsx("form", { onSubmit: (e) => {
                e.preventDefault();
                onSubmit();
            }, children: component })) : (component) }));
});
//# sourceMappingURL=TableFilterInput.js.map