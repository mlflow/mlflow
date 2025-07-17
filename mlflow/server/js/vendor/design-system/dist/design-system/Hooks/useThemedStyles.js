import { useMemo, useRef } from 'react';
import { useDesignSystemTheme } from './useDesignSystemTheme';
/**
 * A helper hook that allows quick creation of theme-dependent styles.
 * Results in more compact code than using useMemo and
 * useDesignSystemTheme separately.
 *
 * @example
 * const styles = useThemedStyles((theme) => ({
 *   overlay: {
 *     backgroundColor: theme.colors.backgroundPrimary,
 *     borderRadius: theme.borders.borderRadiusMd,
 *   },
 *   wrapper: {
 *     display: 'flex',
 *     gap: theme.spacing.md,
 *   },
 * }));

 * <div css={styles.overlay}>...</div>
 *
 * @param styleFactory Factory function that accepts theme object as a parameter and returns
 *     the style object. **Note**: factory function body is being memoized internally and is intended
 *     to be used only for simple style objects that depend solely on the theme. If you want to use
 *     styles that change depending on external values (state, props etc.) you should use
 *     `useDesignSystemTheme` directly with  your own reaction mechanism.
 * @returns The constructed style object
 */
export const useThemedStyles = (styleFactory) => {
    const { theme } = useDesignSystemTheme();
    // We can assume that the factory function won't change and we're
    // observing theme changes only.
    const styleFactoryRef = useRef(styleFactory);
    return useMemo(() => styleFactoryRef.current(theme), [theme]);
};
//# sourceMappingURL=useThemedStyles.js.map