// PatternFly responsive breakpoints override for Databricks Design System
// PatternFly uses similar breakpoints but with slightly different values

type availableBreakpoints = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
type ResponsiveOptions<RType> = Record<availableBreakpoints, RType>;

// PatternFly breakpoints (similar to Bootstrap 4 but with some differences)
const breakpoints: ResponsiveOptions<number> = {
  xs: 0,     // Extra small devices
  sm: 576,   // Small devices (landscape phones)
  md: 768,   // Medium devices (tablets)
  lg: 992,   // Large devices (desktops) 
  xl: 1200,  // Extra large devices (large desktops)
  xxl: 1600, // Extra extra large devices
};

const mediaQueries: ResponsiveOptions<string> = {
  xs: '@media (max-width: 575.98px)',
  sm: `@media (min-width: ${breakpoints.sm}px)`,
  md: `@media (min-width: ${breakpoints.md}px)`,
  lg: `@media (min-width: ${breakpoints.lg}px)`,
  xl: `@media (min-width: ${breakpoints.xl}px)`,
  xxl: `@media (min-width: ${breakpoints.xxl}px)`,
};

export const patternflyResponsive = { breakpoints, mediaQueries };