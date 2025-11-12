import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { TableSkeleton } from './TableSkeleton';
describe('TableSkeleton', () => {
    it('should indicate loading status via aria-busy', () => {
        render(_jsx(TableSkeleton, {}));
        expect(screen.getByRole('status')).toHaveAttribute('aria-busy', 'true');
    });
});
//# sourceMappingURL=TableSkeleton.test.js.map