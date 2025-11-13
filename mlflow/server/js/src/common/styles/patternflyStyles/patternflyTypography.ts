// PatternFly typography tokens override for Databricks Design System
// Based on PatternFly's font size and weight tokens

// PatternFly font sizes (approximated from their token system):
// body--sm: ~12px
// body--default: ~14px  
// body--lg: ~16px
// heading sizes vary from ~18px to ~32px

export const patternflyTypography = {
  // Font sizes
  fontSizeSm: 12,   // --pf-t--global--font--size--body--sm
  fontSizeBase: 14, // --pf-t--global--font--size--body--default (PatternFly's base is slightly larger)
  fontSizeMd: 14,   // same as base
  fontSizeLg: 16,   // --pf-t--global--font--size--body--lg
  fontSizeXl: 20,   // heading sizes
  fontSizeXxl: 28,  // larger heading sizes
  
  // Line heights (PatternFly uses good defaults for readability)
  lineHeightSm: '16px',   // --pf-t--global--font--line-height--body
  lineHeightBase: '20px', // --pf-t--global--font--line-height--body
  lineHeightMd: '20px',   // same as base
  lineHeightLg: '24px',   // --pf-t--global--font--line-height--heading
  lineHeightXl: '28px',   // --pf-t--global--font--line-height--heading
  lineHeightXxl: '36px',  // larger heading line height
  
  // Font weights
  typographyRegularFontWeight: 400, // --pf-t--global--font--weight--body--default
  typographyBoldFontWeight: 700,    // --pf-t--global--font--weight--body--bold (PatternFly uses 700 vs 600)
};