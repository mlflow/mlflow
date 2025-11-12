export type SAFEXValueType = boolean | number;
/**
 * Provides access to http://go/safex flags from the frontend. Note that this is a temporary
 * workaround until direct `safex` imports are available.
 * @param flag The name of the flag to check
 * @param defaultValue The default value to return if the flag is not set
 * @returns
 */
export declare const safex: <T extends SAFEXValueType>(flag: string, defaultValue: T) => T;
//# sourceMappingURL=safex.d.ts.map