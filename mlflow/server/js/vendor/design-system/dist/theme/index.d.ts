import { getBorders, getLegacyBorders } from './borders';
import responsive from './breakpoints';
import { getColors, getProtectedSemanticColors } from './colors';
import generalVariables, { getShadowVariables } from './generalVariables';
import { getGradients } from './gradients';
import { getShadows } from './shadows';
import spacing from './spacing';
import typography from './typography';
export type ComponentTheme = ReturnType<typeof getTheme>;
export interface ThemeOptions {
    enableAnimation: boolean;
    zIndexBase: number;
    useNewBorderColors: boolean;
}
export interface Theme {
    colors: ReturnType<typeof getColors>;
    gradients: ReturnType<typeof getGradients>;
    spacing: typeof spacing;
    general: typeof generalVariables & ReturnType<typeof getShadowVariables>;
    typography: typeof typography;
    shadows: ReturnType<typeof getShadows>;
    /**
     * @deprecated use `borders` instead.
     */
    legacyBorders: ReturnType<typeof getLegacyBorders>;
    borders: ReturnType<typeof getBorders>;
    responsive: typeof responsive;
    isDarkMode: boolean;
    options: ThemeOptions;
    /**
     * @private INTERNAL USE ONLY, DO NOT USE.
     */
    DU_BOIS_INTERNAL_ONLY: {
        colors: ReturnType<typeof getProtectedSemanticColors>;
    };
}
export declare const getTheme: typeof getThemeImpl & import("lodash").MemoizedFunction;
declare function getThemeImpl(isDarkMode: boolean, options?: ThemeOptions): Theme;
export {};
//# sourceMappingURL=index.d.ts.map