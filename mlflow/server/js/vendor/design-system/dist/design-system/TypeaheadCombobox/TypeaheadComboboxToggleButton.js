import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import React from 'react';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronDownIcon } from '../Icon';
const getToggleButtonStyles = (theme, disabled) => {
    return css({
        cursor: 'pointer',
        userSelect: 'none',
        color: theme.colors.textSecondary,
        backgroundColor: 'transparent',
        border: 'none',
        padding: 0,
        marginLeft: theme.spacing.xs,
        height: 16,
        width: 16,
        ...(disabled && {
            pointerEvents: 'none',
            color: theme.colors.actionDisabledText,
        }),
    });
};
export const TypeaheadComboboxToggleButton = React.forwardRef(({ disabled, ...restProps }, ref) => {
    const { theme } = useDesignSystemTheme();
    const { onClick } = restProps;
    function handleClick(e) {
        e.stopPropagation();
        onClick(e);
    }
    return (_jsx("button", { type: "button", "aria-label": "toggle menu", ref: ref, css: getToggleButtonStyles(theme, disabled), ...restProps, onClick: handleClick, children: _jsx(ChevronDownIcon, {}) }));
});
//# sourceMappingURL=TypeaheadComboboxToggleButton.js.map