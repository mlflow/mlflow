import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useDesignSystemTheme } from '../Hooks';
export const Spacer = ({ size = 'md', shrinks, ...props }) => {
    const { theme } = useDesignSystemTheme();
    const spacingValues = {
        xs: theme.spacing.xs,
        sm: theme.spacing.sm,
        md: theme.spacing.md,
        lg: theme.spacing.lg,
    };
    return (_jsx("div", { css: css({
            height: spacingValues[size],
            ...(shrinks === false ? { flexShrink: 0 } : undefined),
        }), ...props }));
};
//# sourceMappingURL=Spacer.js.map