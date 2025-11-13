import { describe, it, expect } from '@jest/globals';
import { getStartAndSuffix } from './TextMiddleElide';
describe('getStartAndSuffix', () => {
    it('should return start and suffix for a long string', () => {
        const { start, suffix } = getStartAndSuffix('organization_name', 6);
        expect(start).toBe('organizatio');
        expect(suffix).toBe('n_name');
    });
    it('should return entire string as start for a shorter than suffix length name', () => {
        const { start, suffix } = getStartAndSuffix('name', 6);
        expect(start).toBe('name');
        expect(suffix).toBeUndefined();
    });
});
//# sourceMappingURL=TextMiddleElide.test.js.map