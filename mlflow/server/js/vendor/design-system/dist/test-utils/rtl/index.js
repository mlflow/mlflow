import userEvent from '@testing-library/user-event';
export * as select from './select';
export * as selectEvent from './selectEvent';
export { simpleSelectTestUtils } from './simpleSelect.utils';
export * from './table';
/**
 * Opens the dropdown menu by clicking on the dropdown button.
 *
 * @param dropdownButton - The Dropdown Trigger button that opens the menu when clicked.
 */
export const openDropdownMenu = async (dropdownButton) => {
    await userEvent.type(dropdownButton, '{arrowdown}');
};
//# sourceMappingURL=index.js.map