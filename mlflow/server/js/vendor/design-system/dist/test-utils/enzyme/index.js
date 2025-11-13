export * as selectEvent from './selectEvent';
export * from './table';
/**
 * Open a dropdown menu by simulating a pointerDown event on the dropdown button.
 *
 * @param dropdownButton - The Dropdown Trigger button that opens the menu when clicked.
 */
export const openDropdownMenu = (dropdownButton) => {
    dropdownButton.hostNodes().simulate('pointerDown', { button: 0, ctrlKey: false });
};
//# sourceMappingURL=index.js.map