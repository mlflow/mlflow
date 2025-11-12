import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const PageWrapper = ({ children, ...props }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { ...addDebugOutlineIfEnabled(), css: css({
            paddingLeft: 16,
            paddingRight: 16,
            backgroundColor: theme.isDarkMode ? theme.colors.backgroundPrimary : 'transparent',
        }), ...props, children: children }));
};
//# sourceMappingURL=PageWrapper.js.map