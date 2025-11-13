// PatternFly spacing tokens override for Databricks Design System
// PatternFly uses rem-based spacing that we need to convert to pixel integers

// PatternFly spacing scale (in pixels, approximated from rem values):
// xs: 0.25rem = 4px
// sm: 0.5rem = 8px  
// md: 1rem = 16px
// lg: 1.5rem = 24px
// xl: 2rem = 32px
// 2xl: 3rem = 48px
// 3xl: 4rem = 64px
// 4xl: 5rem = 80px

export const patternflySpacing = {
  xs: 4,  // --pf-t--global--spacer--xs
  sm: 8,  // --pf-t--global--spacer--sm
  md: 16, // --pf-t--global--spacer--md
  lg: 24, // --pf-t--global--spacer--lg
  xl: 32, // --pf-t--global--spacer--xl
  // Additional PatternFly spacers not in base Databricks spacing
  '2xl': 48, // --pf-t--global--spacer--2xl
  '3xl': 64, // --pf-t--global--spacer--3xl
  '4xl': 80, // --pf-t--global--spacer--4xl
};