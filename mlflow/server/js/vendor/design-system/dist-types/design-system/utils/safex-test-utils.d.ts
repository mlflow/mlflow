import type { SAFEXValueType } from './safex';
type FlagMap = Record<string, SAFEXValueType>;
export type SafexTestingConfig = {
    setSafex: (overrides: FlagMap) => void;
};
export declare function setupSafexTesting(): SafexTestingConfig;
export {};
//# sourceMappingURL=safex-test-utils.d.ts.map