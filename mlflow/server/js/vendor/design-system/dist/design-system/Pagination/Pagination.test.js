import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Pagination } from '.';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
describe('Pagination', () => {
    const onChangeSpy = jest.fn();
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const Component = ({ currentPageIndex }) => (_jsx(DesignSystemEventProviderForTest, { children: _jsx(Pagination, { componentId: "pagination_test", currentPageIndex: currentPageIndex, numTotal: 15, pageSize: 5, onChange: onChangeSpy }) }));
    it('emits value change events when a page number is clicked', async () => {
        render(_jsx(Component, { currentPageIndex: 1 }));
        expect(onChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('listitem', { name: '2' }));
        expect(onChangeSpy).toHaveBeenCalledWith(2, 5);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'pagination_test',
            componentType: 'pagination',
            shouldStartInteraction: false,
            value: 2,
        });
        await userEvent.click(screen.getByRole('listitem', { name: '3' }));
        expect(onChangeSpy).toHaveBeenCalledWith(3, 5);
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'pagination_test',
            componentType: 'pagination',
            shouldStartInteraction: false,
            value: 3,
        });
    });
    it('emits value change event when the next page button is clicked', async () => {
        render(_jsx(Component, { currentPageIndex: 1 }));
        expect(onChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('listitem', { name: 'Next Page' }));
        expect(onChangeSpy).toHaveBeenCalledWith(2, 5);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'pagination_test',
            componentType: 'pagination',
            shouldStartInteraction: false,
            value: 2,
        });
    });
    it('emits value change event when the previous page button is clicked', async () => {
        render(_jsx(Component, { currentPageIndex: 2 }));
        expect(onChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('listitem', { name: 'Previous Page' }));
        expect(onChangeSpy).toHaveBeenCalledWith(1, 5);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'pagination_test',
            componentType: 'pagination',
            shouldStartInteraction: false,
            value: 1,
        });
    });
});
//# sourceMappingURL=Pagination.test.js.map