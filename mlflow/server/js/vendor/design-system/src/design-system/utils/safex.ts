type SAFEXValueType = boolean | number;

/**
 * Provides access to http://go/safex flags from the frontend. Note that this is a temporary
 * workaround until direct `safex` imports are available.
 * @param flag The name of the flag to check
 * @param defaultValue The default value to return if the flag is not set
 * @returns
 */
export const safex: <T extends SAFEXValueType>(flag: string, defaultValue: T) => T = <T>(
  flag: string,
  defaultValue: T,
) => {
  // Catching errors here, because we don't have type-safety to ensure
  // that `__debug__safex`'s API hasn't changed.
  try {
    const globalSafex = (window as any).__debug__safex;
    return globalSafex ? globalSafex(flag, defaultValue) : defaultValue;
  } catch (e) {
    return defaultValue;
  }
};

type FlagMap = Record<string, SAFEXValueType>;

export type SafexTestingConfig = {
  setSafex: (overrides: FlagMap) => void;
};

export function setupSafexTesting(): SafexTestingConfig {
  let flags: FlagMap = {};

  beforeAll(() => {
    const global = window as any;
    global.__debug__safex = (flag: string, defaultValue: SAFEXValueType) => {
      return flags.hasOwnProperty(flag) ? flags[flag] : defaultValue;
    };
  });

  afterAll(() => {
    const global = window as any;
    delete global.__debug__safex;
  });

  beforeEach(() => {
    flags = {};
  });

  function setSafex(overrides: FlagMap) {
    flags = { ...flags, ...overrides };
  }

  return {
    setSafex,
  };
}
