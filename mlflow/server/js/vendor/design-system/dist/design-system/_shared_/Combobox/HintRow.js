import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDesignSystemTheme } from '../../Hooks';
export const HintRow = ({ disabled, children }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: {
            color: theme.colors.textSecondary,
            fontSize: theme.typography.fontSizeSm,
            ...(disabled && {
                color: theme.colors.actionDisabledText,
            }),
        }, children: children }));
};
//# sourceMappingURL=HintRow.js.map