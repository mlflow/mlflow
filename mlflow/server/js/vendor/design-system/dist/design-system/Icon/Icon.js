import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import AntDIcon from '@ant-design/icons';
import { forwardRef, useMemo } from 'react';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useUniqueId } from '../utils/useUniqueId';
const getIconVariantStyles = (theme, color) => {
    switch (color) {
        case 'success':
            return { color: theme.colors.textValidationSuccess };
        case 'warning':
            return { color: theme.colors.textValidationWarning };
        case 'danger':
            return { color: theme.colors.textValidationDanger };
        case 'ai':
            return {
                'svg *': {
                    fill: 'var(--ai-icon-gradient)',
                },
            };
        default:
            return { color: color };
    }
};
export const Icon = forwardRef((props, forwardedRef) => {
    const { component: Component, dangerouslySetAntdProps, color, style, ...otherProps } = props;
    const { theme } = useDesignSystemTheme();
    const linearGradientId = useUniqueId('ai-linear-gradient');
    const MemoizedComponent = useMemo(() => Component
        ? ({ fill, ...iconProps }) => (
        // We don't rely on top-level fills for our colors. Fills are specified
        // with "currentColor" on children of the top-most svg.
        _jsxs(_Fragment, { children: [_jsx(Component, { fill: "none", ...iconProps, style: color === 'ai'
                        ? { ['--ai-icon-gradient']: `url(#${linearGradientId})`, ...iconProps.style }
                        : iconProps.style }), color === 'ai' && (_jsx("svg", { width: "0", height: "0", viewBox: "0 0 0 0", css: visuallyHidden, children: _jsx("defs", { children: _jsxs("linearGradient", { id: linearGradientId, x1: "0%", y1: "0%", x2: "100%", y2: "100%", children: [_jsx("stop", { offset: "20.5%", stopColor: theme.colors.branded.ai.gradientStart }), _jsx("stop", { offset: "46.91%", stopColor: theme.colors.branded.ai.gradientMid }), _jsx("stop", { offset: "79.5%", stopColor: theme.colors.branded.ai.gradientEnd })] }) }) }))] }))
        : undefined, [Component, color, linearGradientId, theme]);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDIcon, { ...addDebugOutlineIfEnabled(), ref: forwardedRef, "aria-hidden": "true", css: {
                fontSize: theme.general.iconFontSize,
                ...getIconVariantStyles(theme, color),
            }, component: MemoizedComponent, style: {
                ...style,
            }, ...otherProps, ...dangerouslySetAntdProps }) }));
});
//# sourceMappingURL=Icon.js.map