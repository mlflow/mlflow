import type { Theme } from '../../theme';
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
export declare const useThemedStyles: <T>(styleFactory: (theme: Theme) => T) => T;
//# sourceMappingURL=useThemedStyles.d.ts.map