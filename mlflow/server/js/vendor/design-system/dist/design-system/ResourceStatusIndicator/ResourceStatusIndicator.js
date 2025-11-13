import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDesignSystemTheme } from '../Hooks';
import { CircleIcon, CircleOffIcon, CircleOutlineIcon } from '../Icon';
const STATUS_TO_ICON = {
    online: ({ theme, style, ...props }) => _jsx(CircleIcon, { color: "success", css: { ...style }, ...props }),
    disconnected: ({ theme, style, ...props }) => (_jsx(CircleOutlineIcon, { css: { color: theme.colors.grey500, ...style }, ...props })),
    offline: ({ theme, style, ...props }) => _jsx(CircleOffIcon, { css: { color: theme.colors.grey500, ...style }, ...props }),
};
export const ResourceStatusIndicator = (props) => {
    const { status, style, ...restProps } = props;
    const { theme } = useDesignSystemTheme();
    const StatusIcon = STATUS_TO_ICON[status];
    return _jsx(StatusIcon, { theme: theme, style: style, ...restProps });
};
//# sourceMappingURL=ResourceStatusIndicator.js.map