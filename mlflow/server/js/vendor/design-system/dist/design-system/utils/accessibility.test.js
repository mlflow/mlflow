import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { visuallyHidden } from './accessibility';
describe('accessibility utilities', () => {
    describe('visuallyHidden', () => {
        it('should still be exposed in the accName and description', () => {
            render(_jsxs(_Fragment, { children: [_jsx("p", { css: visuallyHidden, id: "desc", children: "description" }), _jsx("button", { "aria-describedby": "desc", children: _jsx("span", { css: visuallyHidden, children: "label" }) })] }));
            const button = screen.getByRole('button');
            expect(button).toHaveAccessibleName('label');
            expect(button).toHaveAccessibleDescription('description');
        });
    });
});
//# sourceMappingURL=accessibility.test.js.map