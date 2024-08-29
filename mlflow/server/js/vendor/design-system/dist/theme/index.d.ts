import borders from './borders';
import responsive from './breakpoints';
import { getColors, getProtectedSemanticColors } from './colors';
import generalVariables, { getShadowVariables } from './generalVariables';
import { getGradients } from './gradients';
import spacing from './spacing';
import typography from './typography';
export type ComponentTheme = ReturnType<typeof getTheme>;
export interface ThemeOptions {
    enableAnimation: boolean;
    zIndexBase: number;
}
export interface Theme {
    colors: ReturnType<typeof getColors>;
    gradients: ReturnType<typeof getGradients>;
    spacing: typeof spacing;
    general: typeof generalVariables & ReturnType<typeof getShadowVariables>;
    typography: typeof typography;
    borders: typeof borders;
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
export declare function getTheme(isDarkMode: boolean, options?: ThemeOptions): Theme;
//# sourceMappingURL=index.d.ts.map