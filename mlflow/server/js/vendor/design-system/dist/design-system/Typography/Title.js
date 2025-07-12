import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Typography as AntDTypography } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { getTypographyColor } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const { Title: AntDTitle } = AntDTypography;
function getLevelStyles(theme, props) {
    switch (props.level) {
        case 1:
            return css({
                '&&': {
                    fontSize: theme.typography.fontSizeXxl,
                    lineHeight: theme.typography.lineHeightXxl,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                },
                '& > .anticon': {
                    lineHeight: theme.typography.lineHeightXxl,
                },
            });
        case 2:
            return css({
                '&&': {
                    fontSize: theme.typography.fontSizeXl,
                    lineHeight: theme.typography.lineHeightXl,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                },
                '& > .anticon': {
                    lineHeight: theme.typography.lineHeightXl,
                },
            });
        case 3:
            return css({
                '&&': {
                    fontSize: theme.typography.fontSizeLg,
                    lineHeight: theme.typography.lineHeightLg,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                },
                '& > .anticon': {
                    lineHeight: theme.typography.lineHeightLg,
                },
            });
        case 4:
        default:
            return css({
                '&&': {
                    fontSize: theme.typography.fontSizeMd,
                    lineHeight: theme.typography.lineHeightMd,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                },
                '& > .anticon': {
                    lineHeight: theme.typography.lineHeightMd,
                },
            });
    }
}
function getTitleEmotionStyles(theme, props) {
    return css(getLevelStyles(theme, props), {
        '&&': {
            color: getTypographyColor(theme, props.color, theme.colors.textPrimary),
        },
        '& > .anticon': {
            verticalAlign: 'middle',
        },
    }, props.withoutMargins && {
        '&&': {
            marginTop: '0 !important', // override general styling
            marginBottom: '0 !important', // override general styling
        },
    });
}
export function Title(userProps) {
    const { dangerouslySetAntdProps, withoutMargins, color, elementLevel, ...props } = userProps;
    const { theme } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDTitle, { ...addDebugOutlineIfEnabled(), ...props, level: elementLevel ?? props.level, className: props.className, css: getTitleEmotionStyles(theme, userProps), ...dangerouslySetAntdProps }) }));
}
//# sourceMappingURL=Title.js.map