import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { forwardRef } from 'react';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon } from '../Icon';
export const getSelectedItemStyles = (theme, disabled) => {
    return css({
        backgroundColor: theme.colors.tagDefault,
        borderRadius: theme.general.borderRadiusBase,
        color: theme.colors.textPrimary,
        lineHeight: theme.typography.lineHeightBase,
        fontSize: theme.typography.fontSizeBase,
        marginTop: 2,
        marginBottom: 2,
        marginInlineEnd: theme.spacing.xs,
        paddingRight: 0,
        paddingTop: 0,
        paddingBottom: 0,
        paddingInlineStart: theme.spacing.xs,
        position: 'relative',
        flex: 'none',
        maxWidth: '100%',
        ...(disabled && {
            pointerEvents: 'none',
        }),
    });
};
const getIconContainerStyles = (theme, disabled) => {
    return css({
        width: 16,
        height: 16,
        ':hover': {
            color: theme.colors.actionTertiaryTextHover,
            backgroundColor: theme.colors.tagHover,
        },
        ...(disabled && {
            pointerEvents: 'none',
            color: theme.colors.actionDisabledText,
        }),
    });
};
const getXIconStyles = (theme) => {
    return css({
        fontSize: theme.typography.fontSizeSm,
        verticalAlign: '-1px',
        paddingLeft: theme.spacing.xs / 2,
        paddingRight: theme.spacing.xs / 2,
    });
};
export const TypeaheadComboboxSelectedItem = forwardRef(({ label, item, getSelectedItemProps, removeSelectedItem, disabled, ...restProps }, ref) => {
    const { theme } = useDesignSystemTheme();
    return (_jsxs("span", { ...getSelectedItemProps({ selectedItem: item }), css: getSelectedItemStyles(theme, disabled), ref: ref, ...restProps, children: [_jsx("span", { css: { marginRight: 2, ...(disabled && { color: theme.colors.actionDisabledText }) }, children: label }), _jsx("span", { css: getIconContainerStyles(theme, disabled), children: _jsx(CloseIcon, { "aria-hidden": "false", onClick: (e) => {
                        if (!disabled) {
                            e.stopPropagation();
                            removeSelectedItem(item);
                        }
                    }, css: getXIconStyles(theme), role: "button", "aria-label": "Remove selected item" }) })] }));
});
//# sourceMappingURL=TypeaheadComboboxSelectedItem.js.map