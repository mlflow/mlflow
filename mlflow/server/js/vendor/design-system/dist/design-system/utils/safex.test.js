import { describe, it, expect } from '@jest/globals';
import { safex, setupSafexTesting } from './safex';
describe('Safex testing', () => {
    const { setSafex } = setupSafexTesting();
    it('should return the default value if the flag is not set', () => {
        expect(safex('databricks.fe.safexTest', false)).toBe(false);
    });
    it('should return the correct value for the flag', () => {
        setSafex({ 'databricks.fe.safexTest': true });
        expect(safex('databricks.fe.safexTest', false)).toBe(true);
    });
});
//# sourceMappingURL=safex.test.js.map