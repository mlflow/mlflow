/**
 * Condense our variables into a single object that will be used
 * to build exported `less` configurations.
 */
export declare function getLessVariables(isDarkMode: boolean): Record<string, any>;
/**
 * This is used if you want to pass the theme to `modifyVars`.
 * We use that in `rollup`, but not in `storybook` since we want theme changes
 * to rebuild and refresh in the latter.
 */
export declare function getLessVariablesObject(isDarkMode: boolean): Record<string, string | any>;
/**
 * This exports a synthetic `less` file containing our `less` variables.
 * We use this in storybook so that we can force that environment to refresh
 * as we work on the theme during active development.
 */
export declare function getLessVariablesText(isDarkMode: boolean): string;
//# sourceMappingURL=convertToLessVars.d.ts.map