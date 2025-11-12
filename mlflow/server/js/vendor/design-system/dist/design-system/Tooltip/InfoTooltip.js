import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Tooltip } from './Tooltip';
import { useDesignSystemTheme } from '../Hooks';
import { InfoIcon } from '../Icon';
export const InfoTooltip = ({ content, iconTitle = 'More information', ...props }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(Tooltip, { content: content, ...props, children: _jsx(InfoIcon, { tabIndex: 0, "aria-hidden": "false", "aria-label": iconTitle, alt: iconTitle, css: { color: theme.colors.textSecondary } }) }));
};
//# sourceMappingURL=InfoTooltip.js.map