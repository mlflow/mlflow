import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { getSelectedItemStyles } from './TypeaheadComboboxSelectedItem';
import { useDesignSystemTheme } from '../Hooks';
export const CountBadge = ({ countStartAt, totalCount, disabled }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { css: [
            getSelectedItemStyles(theme),
            { paddingInlineEnd: theme.spacing.xs, ...(disabled && { color: theme.colors.actionDisabledText }) },
        ], children: countStartAt ? `+${totalCount - countStartAt}` : totalCount }));
};
//# sourceMappingURL=CountBadge.js.map