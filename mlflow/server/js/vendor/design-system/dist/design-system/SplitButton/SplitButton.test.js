import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SplitButton } from './SplitButton';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider/DesignSystemProvider';
import { DropdownMenu } from '../DropdownMenu';
describe('SplitButton', () => {
    it('handles clicks', async () => {
        const handlePrimaryClick = jest.fn();
        const handleItemClick = jest.fn();
        const menu = (_jsxs(DropdownMenu.Content, { children: [_jsx(DropdownMenu.Item, { componentId: "SPLIT_BUTTON_HERE.OPTION_1", onClick: handleItemClick, children: "Option 1" }), _jsx(DropdownMenu.Item, { componentId: "SPLIT_BUTTON_HERE.OPTION_2", children: "Option 2" })] }));
        render(_jsx(DesignSystemProvider, { children: _jsx(SplitButton, { componentId: "SPLIT_BUTTON_HERE", onClick: handlePrimaryClick, menu: menu, menuButtonLabel: "MENU_BUTTON", children: "CLICK ME" }) }));
        await userEvent.click(screen.getByText('CLICK ME'));
        expect(handlePrimaryClick).toHaveBeenCalledTimes(1);
        await userEvent.click(screen.getByLabelText('MENU_BUTTON'));
        await userEvent.click(screen.getByText('Option 1'));
        expect(handleItemClick).toHaveBeenCalledTimes(1);
    });
    it('emits analytics events', async () => {
        const handlePrimaryClick = jest.fn();
        const handleItemClick = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        const menu = (_jsxs(DropdownMenu.Content, { children: [_jsx(DropdownMenu.Item, { componentId: "SPLIT_BUTTON_HERE.OPTION_1", onClick: handleItemClick, children: "Option 1" }), _jsx(DropdownMenu.Item, { componentId: "SPLIT_BUTTON_HERE.OPTION_2", children: "Option 2" })] }));
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsx(SplitButton, { componentId: "SPLIT_BUTTON_HERE", onClick: handlePrimaryClick, menu: menu, menuButtonLabel: "MENU_BUTTON", children: "CLICK ME" }) }) }));
        await userEvent.click(screen.getByText('CLICK ME'));
        expect(handlePrimaryClick).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onClick',
            componentId: 'SPLIT_BUTTON_HERE.primary_button',
            componentType: 'button',
            shouldStartInteraction: true,
            isInteractionSubject: true,
            value: undefined,
            event: expect.anything(),
        });
        await userEvent.click(screen.getByLabelText('MENU_BUTTON'));
        await userEvent.click(screen.getByText('Option 1'));
        expect(handleItemClick).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onClick',
            componentId: 'SPLIT_BUTTON_HERE.OPTION_1',
            componentType: 'dropdown_menu_item',
            componentSubType: undefined,
            shouldStartInteraction: true,
            value: undefined,
            event: expect.any(Object),
            isInteractionSubject: true,
        });
    });
});
//# sourceMappingURL=SplitButton.test.js.map