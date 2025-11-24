/**
 * Opens the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already opened or if the menu is unable to be opened.
 */
export declare function openMenu(select: HTMLElement): Promise<void>;
/**
 * Closes the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already closed or if the menu is unable to be closed.
 */
export declare function closeMenu(select: HTMLElement): Promise<void>;
/**
 * Returns a string concatenating the labels for all selected options.
 */
export declare function getLabelText(select: HTMLElement): string;
/**
 * Removes the `option` by clicking its "X" button. Can only be used with a <Select/>
 * component with `mode="multiple"`. The provided strings must match the option label
 * exactly.
 */
export declare function removeMultiSelectOption(select: HTMLElement, option: string): Promise<void>;
/**
 * Selects options from the dropdown menu for a <Select/> component with `mode="multiple"`.
 * The provided strings must match the option labels exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export declare function multiSelect(select: HTMLElement, options: (string | RegExp)[]): Promise<void>;
/**
 * Selects options from the dropdown menu for a <Select/> component without a
 * mode. The provided string must match an option label exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export declare function singleSelect(select: HTMLElement, option: string | RegExp): Promise<void>;
/**
 * Clicks on the "Clear" button. In order for this function to work properly,
 * the `allowClear` prop must be set to `true`.
 */
export declare function clearAll(select: HTMLElement): Promise<void>;
/**
 * Opens the dropdown menu, finds all of the options in the dropdown, closes
 * the menu, and returns a list of the text of each option in order.
 */
export declare function getAllOptions(select: HTMLElement): Promise<string[]>;
/**
 * Creates a new option for a Select with `mode="tags"` by typing it into the input,
 * clicking on the option in the options list, and then closing the menu.
 */
export declare function createNewOption(select: HTMLElement, option: string): Promise<void>;
//# sourceMappingURL=selectEvent.d.ts.map