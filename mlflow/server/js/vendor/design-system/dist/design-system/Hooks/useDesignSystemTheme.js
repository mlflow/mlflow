import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useTheme as useEmotionTheme } from '@emotion/react';
import { getTheme } from '../../theme';
import { useDesignSystemSafexFlags } from '../utils';
export function getClassNamePrefix(theme) {
    const antdThemeName = theme.isDarkMode ? 'dark' : 'light';
    return `${theme.general.classnamePrefix}-${antdThemeName}`;
}
export function getPrefixedClassNameFromTheme(theme, className) {
    return [getClassNamePrefix(theme), className].filter(Boolean).join('-');
}
export function useDesignSystemTheme() {
    const emotionTheme = useEmotionTheme();
    const { useNewBorderColors } = useDesignSystemSafexFlags();
    // Graceful fallback to default theme in case a test or developer forgot context.
    const theme = emotionTheme && emotionTheme.general
        ? emotionTheme
        : getTheme(false, {
            useNewBorderColors,
            // Default values, safe to remove once the border colors flag is retired
            zIndexBase: 1000,
            enableAnimation: false,
        });
    return {
        theme: theme,
        classNamePrefix: getClassNamePrefix(theme),
        getPrefixedClassName: (className) => getPrefixedClassNameFromTheme(theme, className),
    };
}
// This is a simple typed HOC wrapper around the useDesignSystemTheme hook, for use in older react components.
export function WithDesignSystemThemeHoc(Component) {
    return function WrappedWithDesignSystemTheme(props) {
        const themeValues = useDesignSystemTheme();
        return _jsx(Component, { ...props, designSystemThemeApi: themeValues });
    };
}
//# sourceMappingURL=useDesignSystemTheme.js.map