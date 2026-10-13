export type SAFEXValueType = boolean | number;
/**
 * Provides access to http://go/safex flags from the frontend. Note that this is a temporary
 * workaround until direct `safex` imports are available.
 *
 * When the full webapp is loaded, delegates to `window.__debug__safex` which is set by
 * `@databricks/web-shared` and includes full RPC-based flag resolution + local overrides.
 * Falls back to `__TEST_FLAG_OVERRIDES__` for test environments.
 *
 * @param flag The name of the flag to check
 * @param defaultValue The default value to return if the flag is not set
 * @returns
 */
export declare const safex: <T extends SAFEXValueType>(flag: string, defaultValue: T) => T;
/**
 * Reads a server-side feature flag value. Delegates to `window.__debug__serverSideSafe`
 * which is set by `@databricks/web-shared` and includes full flag resolution with
 * local overrides (localStorage + URL params) in dev/staging.
 * Falls back to `__DATABRICKS_SAFE_FLAGS__` and `__TEST_FLAG_OVERRIDES__` when the
 * full webapp implementation is not available.
 *
 * @param flag The name of the flag to check
 * @param defaultValue The default value to return if the flag is not set
 * @returns
 */
export declare const serverSideSafe: <T extends SAFEXValueType>(flag: string, defaultValue: T) => T;
//# sourceMappingURL=safex.d.ts.map