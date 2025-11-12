import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Global, css } from '@emotion/react';
import { useDesignSystemTheme } from '../Hooks';
export const getGlobalStyles = (theme) => {
    return css({
        'body, .mfe-root': {
            backgroundColor: theme.colors.backgroundPrimary,
            color: theme.colors.textPrimary,
            '--dubois-global-background-color': theme.colors.backgroundPrimary,
            '--dubois-global-color': theme.colors.textPrimary,
        },
    });
};
export const ApplyGlobalStyles = () => {
    const { theme } = useDesignSystemTheme();
    return _jsx(Global, { styles: getGlobalStyles(theme) });
};
//# sourceMappingURL=ApplyGlobalStyles.js.map