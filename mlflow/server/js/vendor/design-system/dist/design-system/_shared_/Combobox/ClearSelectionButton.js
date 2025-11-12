import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useDesignSystemTheme } from '../../Hooks';
import { XCircleFillIcon } from '../../Icon';
const getButtonStyles = (theme) => {
    return css({
        color: theme.colors.textPlaceholder,
        fontSize: theme.typography.fontSizeSm,
        marginLeft: theme.spacing.xs,
        ':hover': {
            color: theme.colors.actionTertiaryTextHover,
        },
    });
};
export const ClearSelectionButton = ({ ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(XCircleFillIcon, { "aria-hidden": "false", css: getButtonStyles(theme), role: "button", "aria-label": "Clear selection", ...restProps }));
};
//# sourceMappingURL=ClearSelectionButton.js.map