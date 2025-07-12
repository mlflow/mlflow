import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, beforeEach, jest, afterEach, it, expect } from '@jest/globals';
import { render, act } from '@testing-library/react';
import { LegacyTabs } from './index';
describe('Tabs component', () => {
    let consoleErrorSpy;
    beforeEach(() => {
        consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => { });
    });
    afterEach(() => {
        consoleErrorSpy.mockRestore();
    });
    it('does not throw console warnings about unrecognized getPopupContainer prop', () => {
        act(() => {
            render(_jsx(LegacyTabs, {}));
        });
        expect(consoleErrorSpy).not.toHaveBeenCalledWith(expect.stringMatching(/Warning: React does not recognize the (.*) prop on a DOM element./), 'getPopupContainer', 'getpopupcontainer', expect.stringContaining('at Tabs'));
    });
});
//# sourceMappingURL=LegacyTabs.test.js.map