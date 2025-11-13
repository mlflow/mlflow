import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from './Button';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
describe('Button', () => {
    it('handles clicks', async () => {
        // Arrange
        const handleClick = jest.fn();
        render(_jsx(Button, { componentId: "codegen_design-system_src_design-system_button_button.test.tsx_11", onClick: handleClick }));
        expect(handleClick).not.toHaveBeenCalled();
        // Act
        await userEvent.click(screen.getByRole('button'));
        // Assert
        expect(handleClick).toHaveBeenCalledTimes(1);
    });
    it('handles clicks with DesignSystemEventProvider', async () => {
        // Arrange
        const handleClick = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Button, { componentId: "bestButtonEver", onClick: handleClick }) }));
        expect(handleClick).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        // Act
        await userEvent.click(screen.getByRole('button'));
        // Assert
        expect(handleClick).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onClick',
            componentId: 'bestButtonEver',
            componentType: 'button',
            shouldStartInteraction: true,
            isInteractionSubject: true,
            value: undefined,
            event: expect.anything(),
        });
    });
});
//# sourceMappingURL=Button.test.js.map