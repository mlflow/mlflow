import { queryHelpers, within } from '@testing-library/react';
import { computeAccessibleName } from 'dom-accessibility-api';
function normalizeText(text) {
    return text.replace(/\s+/g, ' ').trim();
}
/**
 * Extracts the display label from a combobox — equivalent to the selected
 * option's label.
 */
export function getDisplayLabel(combobox) {
    return normalizeText(combobox.textContent ?? '');
}
/**
 * Finds the associated listbox for a combobox.
 *
 * @usage
 *
 * ```tsx
 * const combobox = screen.getByRole('combobox', { name: '…' });
 * await userEvent.click(combobox);
 * const listbox = select.getListbox(combobox);
 * await userEvent.click(within(listbox).getByRole('option', { name: '…' }));
 * ```
 */
export function getListbox(combobox) {
    const id = combobox.getAttribute('aria-controls');
    if (!id) {
        throw queryHelpers.getElementError("This doesn't appear to be a combobox. Make sure you're querying the right element: `ByRole('combobox', { name: '…' })`", combobox);
    }
    const listbox = combobox.ownerDocument.getElementById(id);
    if (!listbox) {
        throw queryHelpers.getElementError("Can't find the listbox. Are you sure the select has been opened? `await userEvent.click(combobox)`", combobox.ownerDocument.body);
    }
    return listbox;
}
/**
 * Returns all options associated with a combobox (requires the select to have
 * been opened).
 */
export function getOptions(combobox) {
    const listbox = getListbox(combobox);
    return within(listbox).getAllByRole('option');
}
/**
 * Returns the accessible name for each option in a combobox.
 */
export function getOptionNames(combobox) {
    const options = getOptions(combobox);
    return options.map((option) => computeAccessibleName(option));
}
//# sourceMappingURL=select.js.map