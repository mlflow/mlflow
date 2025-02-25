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

const defaultOptions: ThemeOptions = {
  enableAnimation: false,
  zIndexBase: 1000,
};

// Function to get variables for a certain theme.
// End users should use useDesignSystemTheme instead.
export function getTheme(isDarkMode: boolean, options: ThemeOptions = defaultOptions): Theme {
  return {
    colors: getColors(isDarkMode),
    gradients: getGradients(isDarkMode),
    spacing,
    general: {
      ...generalVariables,
      ...getShadowVariables(isDarkMode),
    },
    shadows: getShadows(isDarkMode),
    typography,
    legacyBorders: getLegacyBorders(),
    // TODO: Update to use `getBorders`
    borders: getBorders(),
    responsive,
    isDarkMode,
    options,
    DU_BOIS_INTERNAL_ONLY: {
      colors: getProtectedSemanticColors(isDarkMode),
    },
  };
}
