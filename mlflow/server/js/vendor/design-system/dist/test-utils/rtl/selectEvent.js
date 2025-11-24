import { queryHelpers, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { selectClasses } from '../common';
/**
 * Allows the helpers in this module to be used when the select element is
 * queried _semantically_ (as if it were a native <select> element) - i.e.
 * `ByRole('combobox', { name: '...' })`, rather than by test ID.
 *
 * Also checks if <DesignSystemProvider> was used, because many of the helpers
 * in this module query by class name starting with "du-bois-", which requires
 * the provider.
 */
function getRootElement(element) {
    if (element.getAttribute('role') === 'combobox') {
        element = element.closest(`.${selectClasses.selector}`).parentElement;
    }
    if (element.classList.contains('ant-select')) {
        throw new Error('Component must be wrapped by <DesignSystemProvider>');
    }
    return element;
}
function getOptionsList(select) {
    const body = select.ownerDocument.body;
    const input = within(select).getByRole('combobox');
    const listId = input.getAttribute('aria-owns') || input.getAttribute('aria-controls');
    if (!listId) {
        throw queryHelpers.getElementError('Options input does not control an options list', body);
    }
    const listbox = select.ownerDocument.getElementById(listId);
    if (!listbox?.parentElement) {
        throw queryHelpers.getElementError('Options listbox does not have a parent', body);
    }
    const optionsList = listbox.parentElement.querySelector(`.${selectClasses.list}`);
    if (!optionsList) {
        throw queryHelpers.getElementError('Options list not found', body);
    }
    return optionsList;
}
/**
 * Opens the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already opened or if the menu is unable to be opened.
 */
export async function openMenu(select) {
    select = getRootElement(select);
    if (select.classList.contains(selectClasses.open)) {
        throw queryHelpers.getElementError('Select is already open', select);
    }
    const selector = select.querySelector(`.${selectClasses.selector}`);
    if (!selector) {
        throw queryHelpers.getElementError('Selector not found', select);
    }
    await userEvent.click(selector, { pointerEventsCheck: 0 });
    await waitFor(() => {
        if (!select.classList.contains(selectClasses.open)) {
            throw queryHelpers.getElementError('Select did not open', select);
        }
    });
}
/**
 * Closes the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already closed or if the menu is unable to be closed.
 */
export async function closeMenu(select) {
    select = getRootElement(select);
    if (!select.classList.contains(selectClasses.open)) {
        throw queryHelpers.getElementError('Select is already closed', select);
    }
    const selector = select.querySelector(`.${selectClasses.selector}`);
    if (!selector) {
        throw queryHelpers.getElementError('Selector not found', select);
    }
    await userEvent.click(selector, { pointerEventsCheck: 0 });
    await waitFor(() => {
        if (select.classList.contains(selectClasses.open)) {
            throw queryHelpers.getElementError('Select did not close', select);
        }
    });
}
/**
 * Returns a string concatenating the labels for all selected options.
 */
export function getLabelText(select) {
    select = getRootElement(select);
    const selector = select.querySelector(`.${selectClasses.selector}`);
    if (!selector) {
        throw queryHelpers.getElementError('Selector not found', select);
    }
    // Trim the text to avoid weird whitespace issues non-label elements being added.
    // For example, the input mirror is an empty span with some whitespace that is
    // nested under the selector but does not show up in the label text.
    return selector.textContent?.trim() ?? '';
}
/**
 * Removes the `option` by clicking its "X" button. Can only be used with a <Select/>
 * component with `mode="multiple"`. The provided strings must match the option label
 * exactly.
 */
export async function removeMultiSelectOption(select, option) {
    select = getRootElement(select);
    const selector = select.querySelector(`.${selectClasses.selector}`);
    if (!selector) {
        throw queryHelpers.getElementError('Selector not found', select);
    }
    const optionItem = within(selector).getByText(option).closest(`.${selectClasses.item}`);
    if (optionItem === null) {
        throw queryHelpers.getElementError(`Option "${option}" not found`, select);
    }
    const removeItem = optionItem.querySelector(`.${selectClasses.removeItem}`);
    if (removeItem === null) {
        throw queryHelpers.getElementError(`Remove button for option "${option}" not found`, optionItem);
    }
    await userEvent.click(removeItem);
}
/**
 * Selects options from the dropdown menu for a <Select/> component with `mode="multiple"`.
 * The provided strings must match the option labels exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export async function multiSelect(select, options) {
    select = getRootElement(select);
    await openMenu(select);
    const optionsList = getOptionsList(select);
    for (let i = 0; i < options.length; i++) {
        const option = options[i];
        const optionItem = within(optionsList).getByText(option);
        await userEvent.click(optionItem, { pointerEventsCheck: 0 });
    }
    // Close the menu to indicate that selection has finished
    await closeMenu(select);
}
/**
 * Selects options from the dropdown menu for a <Select/> component without a
 * mode. The provided string must match an option label exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export async function singleSelect(select, option) {
    select = getRootElement(select);
    await openMenu(select);
    const optionsList = getOptionsList(select);
    const optionItem = within(optionsList).getByText(option);
    await userEvent.click(optionItem, { pointerEventsCheck: 0 });
    // Menu automatically closes for a single <Select/> (no mode="multiple")
}
/**
 * Clicks on the "Clear" button. In order for this function to work properly,
 * the `allowClear` prop must be set to `true`.
 */
export async function clearAll(select) {
    select = getRootElement(select);
    const clearBtn = select.querySelector(`.${selectClasses.clear}`);
    if (!clearBtn) {
        throw queryHelpers.getElementError('Select not clearable', select);
    }
    await userEvent.click(clearBtn);
}
/**
 * Opens the dropdown menu, finds all of the options in the dropdown, closes
 * the menu, and returns a list of the text of each option in order.
 */
export async function getAllOptions(select) {
    select = getRootElement(select);
    await openMenu(select);
    const optionsList = getOptionsList(select);
    const options = [];
    optionsList.querySelectorAll(`.${selectClasses.option}`).forEach((option) => {
        if (option.textContent === null) {
            throw queryHelpers.getElementError('Option had no text content', option);
        }
        options.push(option.textContent);
    });
    await closeMenu(select);
    return options;
}
/**
 * Creates a new option for a Select with `mode="tags"` by typing it into the input,
 * clicking on the option in the options list, and then closing the menu.
 */
export async function createNewOption(select, option) {
    select = getRootElement(select);
    const selectInput = within(select).getByRole('combobox');
    await userEvent.type(selectInput, option);
    const optionsList = getOptionsList(select);
    const optionItem = within(optionsList).getByText(option);
    await userEvent.click(optionItem);
    await closeMenu(select);
}
//# sourceMappingURL=selectEvent.js.map