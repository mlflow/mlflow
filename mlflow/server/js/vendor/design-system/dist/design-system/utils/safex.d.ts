/**
 * Provides access to http://go/safex flags from the frontend. Note that this is a temporary
 * workaround until direct `safex` imports are available.
 * @param flag The name of the flag to check
 * @param defaultValue The default value to return if the flag is not set
 * @returns
 */
export declare const safex: (flag: string, defaultValue: boolean) => any;
type FlagMap = Record<string, boolean>;
export type SafexTestingConfig = {
    setSafex: (overrides: FlagMap) => void;
};
export declare function setupSafexTesting(): SafexTestingConfig;
export {};
//# sourceMappingURL=safex.d.ts.map