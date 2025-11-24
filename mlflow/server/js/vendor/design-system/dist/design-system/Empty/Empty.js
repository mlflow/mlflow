import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useDesignSystemTheme } from '../Hooks';
import { ListIcon } from '../Icon';
import { Typography } from '../Typography';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const { Title, Paragraph } = Typography;
function getEmptyStyles(theme) {
    const styles = {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        maxWidth: 600,
        wordBreak: 'break-word',
        // TODO: This isn't ideal, but migrating to a safer selector would require a SAFE flag / careful migration.
        '> [role="img"]': {
            // Set size of image to 64px
            fontSize: 64,
            color: theme.colors.actionDisabledText,
            marginBottom: theme.spacing.md,
        },
    };
    return css(styles);
}
function getEmptyTitleStyles(theme, clsPrefix) {
    const styles = {
        [`&.${clsPrefix}-typography`]: {
            color: theme.colors.textSecondary,
            marginTop: 0,
            marginBottom: 0,
        },
    };
    return css(styles);
}
function getEmptyDescriptionStyles(theme, clsPrefix) {
    const styles = {
        [`&.${clsPrefix}-typography`]: {
            color: theme.colors.textSecondary,
            marginBottom: theme.spacing.md,
        },
    };
    return css(styles);
}
export const Empty = (props) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { title, description, image = _jsx(ListIcon, {}), button, dangerouslyAppendEmotionCSS, ...dataProps } = props;
    return (_jsx("div", { ...dataProps, ...addDebugOutlineIfEnabled(), css: { display: 'flex', justifyContent: 'center' }, children: _jsxs("div", { css: [getEmptyStyles(theme), dangerouslyAppendEmotionCSS], children: [image, title && (_jsx(Title, { level: 3, css: getEmptyTitleStyles(theme, classNamePrefix), children: title })), _jsx(Paragraph, { css: getEmptyDescriptionStyles(theme, classNamePrefix), children: description }), button] }) }));
};
//# sourceMappingURL=Empty.js.map