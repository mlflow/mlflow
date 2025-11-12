import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CursorPagination } from '.';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
describe('CursorPagination', () => {
    const onPreviousPageSpy = jest.fn();
    const onNextPageSpy = jest.fn();
    const onPageSizeChangeSpy = jest.fn();
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const Component = ({ valueHasNoPii }) => (_jsx(DesignSystemEventProviderForTest, { children: _jsx(CursorPagination, { componentId: "cursor_pagination_test", valueHasNoPii: valueHasNoPii, onPreviousPage: onPreviousPageSpy, onNextPage: onNextPageSpy, hasPreviousPage: true, hasNextPage: true, pageSizeSelect: {
                options: [10, 20, 50],
                default: 10,
                onChange: onPageSizeChangeSpy,
            } }) }));
    it('emits on click event when the next page button is clicked', async () => {
        render(_jsx(Component, {}));
        expect(onPreviousPageSpy).not.toHaveBeenCalled();
        expect(onNextPageSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('button', { name: 'Next' }));
        expect(onNextPageSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'cursor_pagination_test.next_page',
            componentType: 'button',
            shouldStartInteraction: true,
            isInteractionSubject: true,
            value: undefined,
            event: expect.anything(),
        });
    });
    it('emits on click event when the previous page button is clicked', async () => {
        render(_jsx(Component, {}));
        expect(onPreviousPageSpy).not.toHaveBeenCalled();
        expect(onNextPageSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('button', { name: 'Previous' }));
        expect(onPreviousPageSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'cursor_pagination_test.previous_page',
            componentType: 'button',
            shouldStartInteraction: true,
            isInteractionSubject: true,
            value: undefined,
            event: expect.anything(),
        });
    });
    it('emits value change event with value when the page size is changed', async () => {
        render(_jsx(Component, { valueHasNoPii: true }));
        expect(onPageSizeChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('combobox', { name: 'Select page size' }));
        await userEvent.click(screen.getByText('20 / page'));
        expect(onPageSizeChangeSpy).toHaveBeenCalledWith(20);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'cursor_pagination_test.page_size',
            componentType: 'legacy_select',
            shouldStartInteraction: false,
            value: '20',
        });
    });
    it('emits value change event without value when valueHasNoPii is not set', async () => {
        render(_jsx(Component, {}));
        expect(onPageSizeChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('combobox', { name: 'Select page size' }));
        await userEvent.click(screen.getByText('20 / page'));
        expect(onPageSizeChangeSpy).toHaveBeenCalledWith(20);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'cursor_pagination_test.page_size',
            componentType: 'legacy_select',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
});
//# sourceMappingURL=CursorPagination.test.js.map