import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Typography as AntDTypography } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { getTypographyColor } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const { Paragraph: AntDParagraph } = AntDTypography;
function getParagraphEmotionStyles(theme, clsPrefix, props) {
    return css({
        '&&': {
            fontSize: theme.typography.fontSizeBase,
            fontWeight: theme.typography.typographyRegularFontWeight,
            lineHeight: theme.typography.lineHeightBase,
            color: getTypographyColor(theme, props.color, theme.colors.textPrimary),
        },
        '& .anticon': {
            verticalAlign: 'text-bottom',
        },
        [`& .${clsPrefix}-btn-link`]: {
            verticalAlign: 'baseline !important',
        },
    }, props.disabled && { '&&': { color: theme.colors.actionDisabledText } }, props.withoutMargins && {
        '&&': {
            marginTop: 0,
            marginBottom: 0,
        },
    });
}
export function Paragraph(userProps) {
    const { dangerouslySetAntdProps, withoutMargins, color, ...props } = userProps;
    const { theme, classNamePrefix } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDParagraph, { ...addDebugOutlineIfEnabled(), ...props, className: props.className, css: getParagraphEmotionStyles(theme, classNamePrefix, userProps), ...dangerouslySetAntdProps }) }));
}
//# sourceMappingURL=Paragraph.js.map