// PatternFly shadows override for Databricks Design System
// PatternFly has limited shadow tokens, so we keep similar shadow structure but use PatternFly-like values

// PatternFly shadow colors
const darkShadow = 'rgba(0, 0, 0, 0.45)';
const darkShadowLight = 'rgba(0, 0, 0, 0.26)';
const darkShadowHeavy = 'rgba(0, 0, 0, 0.61)';

const lightShadow = 'rgba(0, 0, 0, 0.08)';
const lightShadowLight = 'rgba(0, 0, 0, 0.05)';
const lightShadowHeavy = 'rgba(0, 0, 0, 0.13)';

export const getPatternflyShadows = (isDarkMode: boolean) => {
  return isDarkMode
    ? {
        xs: `0px 1px 0px 0px ${darkShadowLight}`,
        sm: `0px 2px 3px -1px ${darkShadow}, 0px 1px 0px 0px ${darkShadowLight}`,
        md: `0px 3px 6px 0px ${darkShadow}`,
        lg: `0px 2px 16px 0px ${darkShadowHeavy}`,
        xl: `0px 8px 40px 0px ${darkShadowHeavy}`,
      }
    : {
        xs: `0px 1px 0px 0px ${lightShadowLight}`,
        sm: `0px 2px 3px -1px ${lightShadowLight}, 0px 1px 0px 0px ${lightShadow}`,
        md: `0px 3px 6px 0px ${lightShadowLight}`,
        lg: `0px 2px 16px 0px ${lightShadow}`,
        xl: `0px 8px 40px 0px ${lightShadowHeavy}`,
      };
};