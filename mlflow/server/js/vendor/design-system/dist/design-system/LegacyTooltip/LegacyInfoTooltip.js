import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { InfoCircleOutlined } from '@ant-design/icons';
import { LegacyTooltip } from './LegacyTooltip';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';
/**
 * `LegacyInfoTooltip` is deprecated in favor of the new `InfoTooltip` component
 * @deprecated
 */
export const LegacyInfoTooltip = ({ title, tooltipProps, iconTitle, isKeyboardFocusable = true, ...iconProps }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(LegacyTooltip, { useAsLabel: true, title: title, ...tooltipProps, children: _jsx("span", { ...addDebugOutlineIfEnabled(), style: { display: 'inline-flex' }, children: _jsx(InfoCircleOutlined, { tabIndex: isKeyboardFocusable ? 0 : -1, "aria-hidden": "false", "aria-label": iconTitle, alt: iconTitle, css: { fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }, ...iconProps }) }) }));
};
//# sourceMappingURL=LegacyInfoTooltip.js.map