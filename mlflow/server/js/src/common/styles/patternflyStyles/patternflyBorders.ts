// PatternFly border radius tokens override for Databricks Design System
// Based on PatternFly's border radius token system

export const patternflyBorders = {
  // PatternFly border radius tokens:
  borderRadius0: 0,    // --pf-t--global--border--radius--sharp (0px)
  borderRadiusSm: 4,   // --pf-t--global--border--radius--small (4px) 
  borderRadiusMd: 8,   // --pf-t--global--border--radius--medium (8px)
  borderRadiusLg: 12,  // --pf-t--global--border--radius--large (12px)
  borderRadiusFull: 999, // --pf-t--global--border--radius--pill (999px for full rounded)
};

// Legacy borders for compatibility (same as new borders in this case)
export const patternflyLegacyBorders = {
  borderRadiusMd: 4,  // Keep legacy default
  borderRadiusLg: 8,  // Keep legacy large
};