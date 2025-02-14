/**
 * These values are based on AntD's breakpoints which follow BootStrap 4 media query rules.
 * The numerical values represent the min-width of the given size.
 * AntD values: https://ant.design/components/grid#col
 * Bootstrap: https://getbootstrap.com/docs/4.0/layout/overview/#responsive-breakpoints
 */
type availableBreakpoints = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
type ResponsiveOptions<RType> = Record<availableBreakpoints, RType>;

const breakpoints: ResponsiveOptions<number> = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600,
};

const mediaQueries: ResponsiveOptions<string> = {
  xs: '@media (max-width: 575.98px)',
  sm: `@media (min-width: ${breakpoints.sm}px)`,
  md: `@media (min-width: ${breakpoints.md}px)`,
  lg: `@media (min-width: ${breakpoints.lg}px)`,
  xl: `@media (min-width: ${breakpoints.xl}px)`,
  xxl: `@media (min-width: ${breakpoints.xxl}px)`,
};

const responsive = { breakpoints, mediaQueries };
export default responsive;
