/**
 * These values are based on AntD's breakpoints which follow BootStrap 4 media query rules.
 * The numerical values represent the min-width of the given size.
 * AntD values: https://ant.design/components/grid#col
 * Bootstrap: https://getbootstrap.com/docs/4.0/layout/overview/#responsive-breakpoints
 */
type availableBreakpoints = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'xxl';
type ResponsiveOptions<RType> = Record<availableBreakpoints, RType>;
declare const responsive: {
    breakpoints: ResponsiveOptions<number>;
    mediaQueries: ResponsiveOptions<string>;
};
export default responsive;
//# sourceMappingURL=breakpoints.d.ts.map