import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import * as RadixToolbar from '@radix-ui/react-toolbar';
import { forwardRef } from 'react';
import { useDesignSystemSafexFlags, useDesignSystemTheme } from '../../design-system';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
const getRootStyles = (theme, useNewShadows, useNewBorderColors) => {
    return css({
        alignItems: 'center',
        backgroundColor: theme.colors.backgroundSecondary,
        border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusSm,
        boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
        display: 'flex',
        gap: theme.spacing.md,
        width: 'max-content',
        padding: theme.spacing.sm,
    });
};
export const Root = forwardRef((props, ref) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    return (_jsx(RadixToolbar.Root, { ...addDebugOutlineIfEnabled(), css: getRootStyles(theme, useNewShadows, useNewBorderColors), ...props, ref: ref }));
});
export const Button = forwardRef((props, ref) => {
    return _jsx(RadixToolbar.Button, { ...props, ref: ref });
});
const getSeparatorStyles = (theme) => {
    return css({
        alignSelf: 'stretch',
        backgroundColor: theme.colors.borderDecorative,
        width: 1,
    });
};
export const Separator = forwardRef((props, ref) => {
    const { theme } = useDesignSystemTheme();
    return _jsx(RadixToolbar.Separator, { css: getSeparatorStyles(theme), ...props, ref: ref });
});
export const Link = forwardRef((props, ref) => {
    return _jsx(RadixToolbar.Link, { ...props, ref: ref });
});
export const ToggleGroup = forwardRef((props, ref) => {
    return _jsx(RadixToolbar.ToggleGroup, { ...props, ref: ref });
});
const getToggleItemStyles = (theme) => {
    return css({
        background: 'none',
        color: theme.colors.textPrimary,
        border: 'none',
        cursor: 'pointer',
        '&:hover': {
            color: theme.colors.actionDefaultTextHover,
        },
        '&[data-state="on"]': {
            color: theme.colors.actionDefaultTextPress,
        },
    });
};
export const ToggleItem = forwardRef((props, ref) => {
    const { theme } = useDesignSystemTheme();
    return _jsx(RadixToolbar.ToggleItem, { css: getToggleItemStyles(theme), ...props, ref: ref });
});
//# sourceMappingURL=Toolbar.js.map