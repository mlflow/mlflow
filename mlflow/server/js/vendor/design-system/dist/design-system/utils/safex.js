/**
 * Provides access to http://go/safex flags from the frontend. Note that this is a temporary
 * workaround until direct `safex` imports are available.
 * @param flag The name of the flag to check
 * @param defaultValue The default value to return if the flag is not set
 * @returns
 */
export const safex = (flag, defaultValue) => {
    // Catching errors here, because we don't have type-safety to ensure
    // that `__debug__safex`'s API hasn't changed.
    try {
        const globalSafex = window.__debug__safex;
        return globalSafex ? globalSafex(flag, defaultValue) : defaultValue;
    }
    catch (e) {
        return defaultValue;
    }
};
export function setupSafexTesting() {
    let flags = {};
    beforeAll(() => {
        const global = window;
        global.__debug__safex = (flag, defaultValue) => {
            return flags.hasOwnProperty(flag) ? flags[flag] : defaultValue;
        };
    });
    afterAll(() => {
        const global = window;
        delete global.__debug__safex;
    });
    beforeEach(() => {
        flags = {};
    });
    function setSafex(overrides) {
        flags = { ...flags, ...overrides };
    }
    return {
        setSafex,
    };
}
//# sourceMappingURL=safex.js.map