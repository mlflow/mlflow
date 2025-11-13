import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, jest, beforeEach, it, expect } from '@jest/globals';
import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Listbox } from './index';
import { DesignSystemEventProvider, DesignSystemEventProviderAnalyticsEventTypes, } from '../../design-system/DesignSystemEventProvider';
import { DesignSystemProvider } from '../../design-system/DesignSystemProvider';
const options = [
    { value: 'apple', label: 'Apple' },
    { value: 'banana', label: 'Banana' },
    { value: 'orange', label: 'Orange' },
];
const eventCallback = jest.fn();
const renderWithProvider = (ui) => {
    return render(_jsx(DesignSystemEventProvider, { callback: eventCallback, children: _jsx(DesignSystemProvider, { children: ui }) }));
};
describe('Listbox', () => {
    const handleSelect = jest.fn();
    beforeEach(() => {
        handleSelect.mockClear();
    });
    it('renders all options', () => {
        renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, initialSelectedValue: "banana", "aria-label": "Test listbox", componentId: "listbox_test" }));
        const listbox = screen.getByRole('listbox');
        expect(listbox).toBeInTheDocument();
        options.forEach((option) => {
            expect(screen.getByRole('option', { name: option.label })).toBeInTheDocument();
        });
        // banana should be selected initially
        const banana = screen.getByRole('option', { name: 'Banana' });
        expect(banana).toHaveAttribute('aria-selected', 'true');
    });
    it('calls onSelect and emits value change event when clicking an option', async () => {
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
        renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, "aria-label": "Test listbox", componentId: "listbox_test", initialSelectedValue: "apple", valueHasNoPii: true, analyticsEvents: [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ] }));
        await waitFor(() => {
            expect(screen.getByText('Banana')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'listbox_test',
            componentType: 'listbox',
            componentSubType: undefined,
            componentViewId: 'testuuid-00v4-0000-0000-000000000000',
            shouldStartInteraction: false,
            value: 'apple',
        });
        await userEvent.click(screen.getByRole('option', { name: 'Banana' }));
        expect(handleSelect).toHaveBeenCalledWith('banana');
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'listbox_test',
            componentType: 'listbox',
            componentSubType: undefined,
            componentViewId: 'testuuid-00v4-0000-0000-000000000000',
            shouldStartInteraction: false,
            value: 'banana',
        });
    });
    describe('keyboard navigation', () => {
        it('allows navigation with arrow keys', async () => {
            renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, "aria-label": "Test listbox", componentId: "listbox_test" }));
            const listbox = screen.getByRole('listbox');
            const apple = screen.getByRole('option', { name: 'Apple' });
            const banana = screen.getByRole('option', { name: 'Banana' });
            const orange = screen.getByRole('option', { name: 'Orange' });
            // Focus the listbox to initialize keyboard navigation
            await userEvent.click(listbox);
            // No option should be highlighted initially
            expect(apple).not.toHaveAttribute('data-highlighted', 'true');
            // Press arrow down
            await userEvent.keyboard('{ArrowDown}');
            expect(apple).toHaveAttribute('data-highlighted', 'true');
            // Press arrow down
            await userEvent.keyboard('{ArrowDown}');
            expect(banana).toHaveAttribute('data-highlighted', 'true');
            // Press arrow up
            await userEvent.keyboard('{ArrowUp}');
            expect(apple).toHaveAttribute('data-highlighted', 'true');
            // Press End
            await userEvent.keyboard('{End}');
            expect(orange).toHaveAttribute('data-highlighted', 'true');
            // Press Home
            await userEvent.keyboard('{Home}');
            expect(apple).toHaveAttribute('data-highlighted', 'true');
        });
        it('selects option with Enter key when option does not have link type', async () => {
            renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, "aria-label": "Test listbox", componentId: "listbox_test" }));
            const listbox = screen.getByRole('listbox');
            // Focus and initialize keyboard navigation
            await userEvent.click(listbox);
            // Navigate to Banana
            await userEvent.keyboard('{ArrowDown}');
            await userEvent.keyboard('{ArrowDown}');
            const banana = screen.getByRole('option', { name: 'Banana' });
            expect(banana).toHaveAttribute('data-highlighted', 'true');
            // Select Banana
            await userEvent.keyboard('{Enter}');
            expect(handleSelect).toHaveBeenCalledWith('banana');
            expect(banana).toHaveAttribute('aria-selected', 'true');
        });
        describe('with link type options', () => {
            const optionsWithLink = [
                {
                    value: 'apple',
                    label: 'Apple',
                    renderOption: (additionalProps) => _jsx("a", { ...additionalProps, children: "Apple" }),
                    href: 'https://example.com',
                },
            ];
            it('does not select option and opens link with Enter key when option has an href', async () => {
                renderWithProvider(_jsx(Listbox, { options: optionsWithLink, onSelect: handleSelect, "aria-label": "Test listbox", componentId: "listbox_test" }));
                const listbox = screen.getByRole('listbox');
                // Focus and initialize keyboard navigation
                await userEvent.click(listbox);
                const openSpy = jest.spyOn(window, 'open').mockImplementation(() => null);
                // Press arrow down once to highlight Apple
                await userEvent.keyboard('{ArrowDown}');
                await userEvent.keyboard('{Enter}');
                expect(handleSelect).toHaveBeenCalledWith('apple');
                const apple = screen.getByRole('link', { name: 'Apple' });
                expect(apple).toHaveAttribute('aria-selected', 'false');
                // Check if the window opens the link
                expect(openSpy).toHaveBeenCalledWith('https://example.com', '_blank');
                openSpy.mockRestore();
            });
            it('does not select option and opens link when clicking an option with an href', async () => {
                renderWithProvider(_jsx(Listbox, { options: optionsWithLink, onSelect: handleSelect, "aria-label": "Test listbox", componentId: "listbox_test" }));
                const openSpy = jest.spyOn(window, 'open').mockImplementation(() => null);
                const apple = screen.getByRole('link', { name: 'Apple' });
                await userEvent.click(apple);
                expect(handleSelect).toHaveBeenCalledWith('apple');
                expect(apple).toHaveAttribute('aria-selected', 'false');
                // Check if the window opens the link
                expect(openSpy).toHaveBeenCalledWith('https://example.com', '_blank');
                openSpy.mockRestore();
            });
        });
    });
    describe('with filter input', () => {
        it('renders filter input when includeFilterInput is true', () => {
            renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, includeFilterInput: true, filterInputPlaceholder: "Search...", "aria-label": "Test listbox", componentId: "listbox_test" }));
            expect(screen.getByRole('combobox')).toBeInTheDocument();
            expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument();
        });
        it('filters options based on input', async () => {
            renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, includeFilterInput: true, filterInputPlaceholder: "Search...", "aria-label": "Test listbox", componentId: "listbox_test" }));
            const input = screen.getByRole('combobox');
            await userEvent.type(input, 'ban');
            const listbox = screen.getByRole('listbox');
            const visibleOptions = within(listbox).getAllByRole('option');
            expect(visibleOptions).toHaveLength(1);
            expect(visibleOptions[0]).toHaveTextContent('Banana');
        });
        it('shows empty message when no options match filter input', async () => {
            renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, includeFilterInput: true, filterInputEmptyMessage: "No results found", filterInputPlaceholder: "Search...", "aria-label": "Test listbox", componentId: "listbox_test" }));
            const input = screen.getByRole('combobox');
            await userEvent.type(input, 'can');
            const emptyMessage = screen.getByRole('status');
            expect(emptyMessage).toHaveTextContent('No results found');
        });
        it('allows keyboard navigation in filtered results', async () => {
            renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, includeFilterInput: true, filterInputPlaceholder: "Search...", "aria-label": "Test listbox", componentId: "listbox_test" }));
            const input = screen.getByRole('combobox');
            await userEvent.type(input, 'a'); // Should show Apple, Banana, Orange
            // Press arrow down twice to highlight Banana
            await userEvent.keyboard('{ArrowDown}');
            await userEvent.keyboard('{ArrowDown}');
            const banana = screen.getByRole('option', { name: 'Banana' });
            expect(banana).toHaveAttribute('data-highlighted', 'true');
            await userEvent.keyboard('{Enter}');
            expect(handleSelect).toHaveBeenCalledWith('banana');
        });
        it('clears filter input with clear button', async () => {
            renderWithProvider(_jsx(Listbox, { options: options, onSelect: handleSelect, includeFilterInput: true, filterInputPlaceholder: "Search...", "aria-label": "Test listbox", componentId: "listbox_test" }));
            const input = screen.getByRole('combobox');
            await userEvent.type(input, 'ban');
            // Find and click the clear button (it has aria-label="close-circle")
            const clearButton = screen.getByRole('button', { name: 'close-circle' });
            await userEvent.click(clearButton);
            expect(input).toHaveValue('');
            // All options should be visible again
            const listbox = screen.getByRole('listbox');
            const visibleOptions = within(listbox).getAllByRole('option');
            expect(visibleOptions).toHaveLength(options.length);
        });
    });
});
//# sourceMappingURL=Listbox.test.js.map