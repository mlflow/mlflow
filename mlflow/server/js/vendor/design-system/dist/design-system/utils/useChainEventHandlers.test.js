import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { render, renderHook } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useChainEventHandlers } from './useChainEventHandlers';
describe('useChainEventHandlers', () => {
    it('should invoke callback with a single hander', async () => {
        // Arrange
        const callback = jest.fn();
        const { result: onKeyDown } = renderHook(() => useChainEventHandlers({ handlers: [callback] }));
        const screen = render(_jsx("input", { onKeyDown: onKeyDown.current }));
        // Act
        await userEvent.type(screen.getByRole('textbox'), '{Enter}');
        // Assert
        expect(callback).toHaveBeenCalledTimes(1);
    });
    it('should invoke callback with multiple handlers', async () => {
        // Arrange
        const callback1 = jest.fn();
        const callback2 = jest.fn();
        const { result: onKeyDown } = renderHook(() => useChainEventHandlers({ handlers: [callback1, callback2] }));
        const screen = render(_jsx("input", { onKeyDown: onKeyDown.current }));
        // Act
        await userEvent.type(screen.getByRole('textbox'), '{Enter}');
        // Assert
        expect(callback1).toHaveBeenCalledTimes(1);
        expect(callback2).toHaveBeenCalledTimes(1);
    });
    it('should invoke callback if event is defaultPrevented and stopOnDefaultPrevented off', async () => {
        // Arrange
        const callback1 = jest.fn((e) => e.preventDefault());
        const callback2 = jest.fn();
        const { result: onKeyDown } = renderHook(() => useChainEventHandlers({ handlers: [callback1, callback2], stopOnDefaultPrevented: false }));
        const screen = render(_jsx("input", { onKeyDown: onKeyDown.current }));
        // Act
        await userEvent.type(screen.getByRole('textbox'), '{Enter}');
        // Assert
        expect(callback1).toHaveBeenCalledTimes(1);
        expect(callback2).toHaveBeenCalledTimes(1);
    });
    it('should not invoke callback if event is defaultPrevented and stopOnDefaultPrevented on', async () => {
        // Arrange
        const callback1 = jest.fn((e) => e.preventDefault());
        const callback2 = jest.fn();
        const { result: onKeyDown } = renderHook(() => useChainEventHandlers({ handlers: [callback1, callback2], stopOnDefaultPrevented: true }));
        const screen = render(_jsx("input", { onKeyDown: onKeyDown.current }));
        // Act
        await userEvent.type(screen.getByRole('textbox'), '{Enter}');
        // Assert
        expect(callback1).toHaveBeenCalledTimes(1);
        expect(callback2).not.toHaveBeenCalled();
    });
});
//# sourceMappingURL=useChainEventHandlers.test.js.map