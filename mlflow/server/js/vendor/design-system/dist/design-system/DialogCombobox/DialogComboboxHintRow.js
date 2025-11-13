import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDesignSystemTheme } from '../Hooks';
export const DialogComboboxHintRow = ({ children }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: {
            minWidth: '100%',
            color: theme.colors.textSecondary,
            fontSize: theme.typography.fontSizeSm,
            '[data-disabled] &': {
                color: theme.colors.actionDisabledText,
            },
        }, children: children }));
};
//# sourceMappingURL=DialogComboboxHintRow.js.map