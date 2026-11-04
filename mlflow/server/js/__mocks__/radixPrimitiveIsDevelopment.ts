// Newer Radix packages import `@radix-ui/primitive/is-development`, a subpath that exists only in the
// package's "exports" map, which Jest 27's resolver doesn't read. This stands in with the value the
// package exports under its default (production) condition.
export const IS_DEVELOPMENT = false;
