import { findByRole, findByText, waitFor } from './utils';
import { selectClasses } from '../common';
/**
 * Clicks on the "Clear" button. In order for this function to work properly,
 * the `allowClear` prop must be set to `true`.
 */
export function clearAll(getSelect) {
    getSelect().find(`.${selectClasses.clear}`).hostNodes().simulate('mousedown');
}
/**
 * Closes the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already closed or if the menu is unable to be closed.
 */
export async function closeMenu(getSelect) {
    if (!getSelect().find(`.${selectClasses.open}`).exists()) {
        throw new Error(`Select is already closed\n${getSelect().debug()}`);
    }
    getSelect().find(`.${selectClasses.selector}`).simulate('mousedown');
    await waitFor(() => {
        const select = getSelect();
        if (select.find(`.${selectClasses.open}`).exists()) {
            throw new Error(`Select did not close\n${select.debug()}`);
        }
    });
}
/**
 * Returns a string concatenating the labels for all selected options.
 */
export function getLabelText(getSelect) {
    // Trim the text to avoid weird whitespace issues non-label elements being added.
    // For example, the input mirror is an empty span with some whitespace that is
    // nested under the selector but does not show up in the label text.
    return getSelect().find(`.${selectClasses.selector}`).text().trim();
}
/**
 * Removes the `option` by clicking its "X" button. Can only be used with a <Select/>
 * component with `mode="multiple"`. The provided strings must match the option label
 * exactly.
 */
export function removeMultiSelectOption(getSelect, option) {
    const optionItem = findByText(getSelect().find(`.${selectClasses.selector}`), option).closest(`.${selectClasses.item}`);
    const removeItem = optionItem.find(`.${selectClasses.removeItem}`).hostNodes();
    removeItem.simulate('click');
}
/**
 * Selects options from the dropdown menu for a <Select/> component with `mode="multiple"`.
 * The provided strings must match the option labels exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export async function multiSelect(getSelect, options) {
    await openMenu(getSelect);
    options.forEach((option) => {
        findByText(getSelect().find(`.${selectClasses.list}`), option).simulate('click');
    });
    // Close the menu to indicate that selection has finished
    await closeMenu(getSelect);
}
/**
 * Selects options from the dropdown menu for a <Select/> component without a
 * mode. The provided string must match an option label exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export async function singleSelect(getSelect, option) {
    await openMenu(getSelect);
    findByText(getSelect().find(`.${selectClasses.list}`), option).simulate('click');
    // Menu automatically closes for a single <Select/> (no mode="multiple")
}
/**
 * Opens the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already opened or if the menu is unable to be opened.
 */
export async function openMenu(getSelect) {
    if (getSelect().find(`.${selectClasses.open}`).exists()) {
        throw new Error(`Select is already open\n${getSelect().debug()}`);
    }
    getSelect().find(`.${selectClasses.selector}`).simulate('mousedown');
    await waitFor(() => {
        const select = getSelect();
        if (!select.find(`.${selectClasses.open}`).exists()) {
            throw new Error(`Select did not open\n${select.debug()}`);
        }
    });
}
/**
 * Opens the dropdown menu, finds all of the options in the dropdown, closes
 * the menu, and returns a list of the text of each option in order.
 */
export async function getAllOptions(getSelect) {
    await openMenu(getSelect);
    const options = getSelect()
        .find(`.${selectClasses.list}`)
        .find(`.${selectClasses.option}`)
        .map((option) => option.text());
    await closeMenu(getSelect);
    return options;
}
/**
 * Creates a new option for a Select with `mode="tags"` by typing it into the input,
 * clicking on the option in the options list, and then closing the menu.
 */
export async function createNewOption(getSelect, option) {
    const selectInput = findByRole(getSelect(), 'combobox');
    selectInput.simulate('change', { target: { value: option } });
    const optionList = getSelect().find(`.${selectClasses.list}`);
    const optionItem = findByText(optionList, option);
    optionItem.simulate('click');
    await closeMenu(getSelect);
}
//# sourceMappingURL=selectEvent.js.map