import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import * as RadixNavigationMenu from '@radix-ui/react-navigation-menu';
import React from 'react';
import { useDesignSystemTheme } from '..';
import { getCommonTabsListStyles, getCommonTabsTriggerStyles } from '../_shared_';
export const Root = React.forwardRef((props, forwardedRef) => {
    return _jsx(RadixNavigationMenu.Root, { ...props, ref: forwardedRef });
});
export const List = React.forwardRef((props, forwardedRef) => {
    const { theme } = useDesignSystemTheme();
    const commonTabsListStyles = getCommonTabsListStyles(theme);
    return (_jsx(RadixNavigationMenu.List, { css: {
            ...commonTabsListStyles,
            marginTop: 0,
            padding: 0,
            overflow: 'auto hidden',
            listStyle: 'none',
        }, ...props, ref: forwardedRef }));
});
export const Item = React.forwardRef(({ children, active, ...props }, forwardedRef) => {
    const { theme } = useDesignSystemTheme();
    const commonTabsTriggerStyles = getCommonTabsTriggerStyles(theme);
    return (_jsx(RadixNavigationMenu.Item, { css: {
            ...commonTabsTriggerStyles,
            height: theme.general.heightSm,
            minWidth: theme.spacing.lg,
            justifyContent: 'center',
            ...(active && {
                // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
                // jumping when switching tabs.
                boxShadow: `inset 0 -4px 0 ${theme.colors.actionPrimaryBackgroundDefault}`,
            }),
        }, ...props, ref: forwardedRef, children: _jsx(RadixNavigationMenu.Link, { asChild: true, active: active, css: {
                padding: `${theme.spacing.xs}px 0 ${theme.spacing.sm}px 0`,
                '&:focus': {
                    outline: `2px auto ${theme.colors.actionDefaultBorderFocus}`,
                    outlineOffset: '-1px',
                },
                '&&': {
                    color: active ? theme.colors.textPrimary : theme.colors.textSecondary,
                    textDecoration: 'none',
                    '&:hover': {
                        color: active ? theme.colors.textPrimary : theme.colors.actionDefaultTextHover,
                        textDecoration: 'none',
                    },
                    '&:focus': {
                        textDecoration: 'none',
                    },
                    '&:active': {
                        color: active ? theme.colors.textPrimary : theme.colors.actionDefaultTextPress,
                    },
                },
            }, children: children }) }));
});
//# sourceMappingURL=NavigationMenu.js.map