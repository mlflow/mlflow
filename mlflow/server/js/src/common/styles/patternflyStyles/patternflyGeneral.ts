// PatternFly general variables override for Databricks Design System
// Note: Many AntD-specific variables don't have PatternFly equivalents, so we skip those

export const patternflyGeneral = {
  // AntD specific properties (required by the type)
  classnamePrefix: 'ant',
  iconfontCssPrefix: 'anticon',
  
  // PatternFly uses slightly different border radius values
  borderRadiusBase: 4, // PatternFly default border radius
  borderWidth: 1, // Standard border width
  
  // Icon sizes - PatternFly uses similar icon sizing
  iconSize: 24,
  iconFontSize: 16,
  
  // Heights - keeping reasonable defaults since PatternFly heights are contextual
  heightSm: 32,
  heightBase: 40,
  buttonHeight: 40,
  buttonInnerHeight: 24, // heightBase - padding - borders
};

// Shadow RGB values for PatternFly theme
export const shadowLightRgb = '31, 39, 45' as const;
export const shadowDarkRgb = '0, 0, 0' as const;

export const getPatternflyShadowVariables = (isDarkMode: boolean) => {
  if (isDarkMode) {
    return {
      shadowLow: `0px 4px 16px rgba(${shadowDarkRgb}, 0.12)`,
      shadowHigh: `0px 8px 24px rgba(${shadowDarkRgb}, 0.2);`,
    } as const;
  } else {
    return {
      shadowLow: `0px 4px 16px rgba(${shadowLightRgb}, 0.12)`,
      shadowHigh: `0px 8px 24px rgba(${shadowLightRgb}, 0.2)`,
    } as const;
  }
};