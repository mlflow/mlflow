import { Fragment as _Fragment, jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { render } from '@testing-library/react';
import { DesignSystemThemeProvider, DesignSystemProvider } from './DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
describe('DesignSystemThemeProvider', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it.each([{ value: true }, { value: false }])('sets the dark mode value to $value in the context', ({ value }) => {
        const TestComponent = () => {
            const { theme } = useDesignSystemTheme();
            expect(theme.isDarkMode).toBe(value);
            return _jsx(_Fragment, {});
        };
        render(
        // eslint-disable-next-line react/forbid-elements
        _jsx(DesignSystemThemeProvider, { isDarkMode: value, children: _jsx(DesignSystemProvider, { children: _jsx(TestComponent, {}) }) }));
    });
});
//# sourceMappingURL=DesignSystemThemeProvider.test.js.map