import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { fireEvent, render, renderHook } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useCallbackOnEnter } from './useCallbackOnEnter';
describe('useCallbackOnEnter', () => {
    it('should call callback when Enter is pressed', async () => {
        // Arrange
        const callback = jest.fn();
        setupTest({ callback, allowBasicEnter: true, allowPlatformEnter: true });
        // Act
        await userEvent.keyboard('{Enter}');
        // Assert
        expect(callback).toHaveBeenCalledTimes(1);
    });
    it('should not call callback when another key is pressed', async () => {
        // Arrange
        const callback = jest.fn();
        setupTest({ callback, allowBasicEnter: true, allowPlatformEnter: true });
        // Act
        await userEvent.keyboard('a');
        // Assert
        expect(callback).not.toHaveBeenCalled();
    });
    it('should not call callback when Enter is pressed with shift key', async () => {
        // Arrange
        const callback = jest.fn();
        setupTest({ callback, allowBasicEnter: true, allowPlatformEnter: true });
        // Act
        await userEvent.keyboard('{Shift>}{Enter}');
        // Assert
        expect(callback).not.toHaveBeenCalled();
    });
    it('should not call callback when composing text and Enter is pressed', async () => {
        // Arrange
        const callback = jest.fn();
        const { input } = setupTest({ callback, allowBasicEnter: true, allowPlatformEnter: true });
        // Act
        await fireEvent.compositionStart(input, { type: 'compositionstart' });
        await userEvent.keyboard('{Enter}');
        // Assert
        expect(callback).not.toHaveBeenCalled();
    });
    it('should call callback when composing text is ended and Enter is pressed', async () => {
        // Arrange
        const callback = jest.fn();
        const { input } = setupTest({ callback, allowBasicEnter: true, allowPlatformEnter: true });
        // Act
        await fireEvent.compositionStart(input, { type: 'compositionstart' });
        await fireEvent.compositionEnd(input, { type: 'compositionend' });
        await userEvent.keyboard('{Enter}');
        // Assert
        expect(callback).toHaveBeenCalledTimes(1);
    });
    /**
     * Instrumentation to setup the test.
     * 1. Renders the useSubmitOnEnter hook
     * 2. Renders an input element with the hook's props
     * 3. Focuses the input element
     * returns the input element for further interaction
     */
    const setupTest = (...parameters) => {
        const { result } = renderHook(() => useCallbackOnEnter(...parameters));
        const screen = render(_jsx("input", { ...result.current }));
        const input = screen.getByRole('textbox');
        input.focus();
        return { input };
    };
});
//# sourceMappingURL=useCallbackOnEnter.test.js.map