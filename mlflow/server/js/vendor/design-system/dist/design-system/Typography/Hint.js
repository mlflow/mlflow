import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Typography as AntDTypography } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const { Text: AntDText } = AntDTypography;
function getTextEmotionStyles(theme, props) {
    return css({
        '&&': {
            display: 'block',
            fontSize: theme.typography.fontSizeSm,
            lineHeight: theme.typography.lineHeightSm,
            color: theme.colors.textSecondary,
            ...(props.withoutMargins && {
                '&&': {
                    marginTop: 0,
                    marginBottom: 0,
                },
            }),
        },
    });
}
export function Hint(userProps) {
    const { dangerouslySetAntdProps, bold, withoutMargins, color, ...props } = userProps;
    const { theme } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDText, { ...addDebugOutlineIfEnabled(), ...props, css: getTextEmotionStyles(theme, userProps), ...dangerouslySetAntdProps }) }));
}
//# sourceMappingURL=Hint.js.map