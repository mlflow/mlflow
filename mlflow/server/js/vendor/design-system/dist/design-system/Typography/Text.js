import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Typography as AntDTypography } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { getTypographyColor } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const { Text: AntDText } = AntDTypography;
function getTextEmotionStyles(theme, props) {
    return css({
        '&&': {
            fontSize: theme.typography.fontSizeBase,
            fontWeight: theme.typography.typographyRegularFontWeight,
            lineHeight: theme.typography.lineHeightBase,
            color: getTypographyColor(theme, props.color, theme.colors.textPrimary),
        },
    }, props.disabled && { '&&': { color: theme.colors.actionDisabledText } }, props.hint && {
        '&&': { fontSize: theme.typography.fontSizeSm, lineHeight: theme.typography.lineHeightSm },
    }, props.bold && {
        '&&': {
            fontSize: theme.typography.fontSizeBase,
            fontWeight: theme.typography.typographyBoldFontWeight,
            lineHeight: theme.typography.lineHeightBase,
        },
    }, props.code && {
        '&& > code': {
            color: theme.colors.textPrimary,
            fontSize: theme.typography.fontSizeBase,
            lineHeight: theme.typography.lineHeightBase,
            background: theme.colors.typographyCodeBg,
            fontFamily: 'monospace',
            borderRadius: theme.borders.borderRadiusSm,
            padding: '2px 4px',
            border: 'unset',
            margin: 0,
        },
    }, props.size && {
        '&&': (() => {
            switch (props.size) {
                case 'xxl':
                    return {
                        fontSize: theme.typography.fontSizeXxl,
                        lineHeight: theme.typography.lineHeightXxl,
                        '& .anticon': {
                            lineHeight: theme.typography.lineHeightXxl,
                            verticalAlign: 'middle',
                        },
                    };
                case 'xl':
                    return {
                        fontSize: theme.typography.fontSizeXl,
                        lineHeight: theme.typography.lineHeightXl,
                        '& .anticon': {
                            lineHeight: theme.typography.lineHeightXl,
                            verticalAlign: 'middle',
                        },
                    };
                case 'lg':
                    return {
                        fontSize: theme.typography.fontSizeLg,
                        lineHeight: theme.typography.lineHeightLg,
                        '& .anticon': {
                            lineHeight: theme.typography.lineHeightLg,
                            verticalAlign: 'middle',
                        },
                    };
                case 'sm':
                    return {
                        fontSize: theme.typography.fontSizeSm,
                        lineHeight: theme.typography.lineHeightSm,
                        '& .anticon': {
                            verticalAlign: '-0.219em',
                        },
                    };
                default:
                    return {};
            }
        })(),
    }, props.withoutMargins && {
        '&&': {
            marginTop: 0,
            marginBottom: 0,
        },
    });
}
export function Text(userProps) {
    // Omit props that are not supported by `antd`
    const { dangerouslySetAntdProps, bold, hint, withoutMargins, color, ...props } = userProps;
    const { theme } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDText, { ...addDebugOutlineIfEnabled(), ...props, className: props.className, css: getTextEmotionStyles(theme, userProps), ...dangerouslySetAntdProps }) }));
}
//# sourceMappingURL=Text.js.map