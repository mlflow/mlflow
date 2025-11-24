import type { ReactWrapper } from 'enzyme';
/**
 * Clicks on the "Clear" button. In order for this function to work properly,
 * the `allowClear` prop must be set to `true`.
 */
export declare function clearAll<P, S, C>(getSelect: () => ReactWrapper<P, S, C>): void;
/**
 * Closes the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already closed or if the menu is unable to be closed.
 */
export declare function closeMenu<P, S, C>(getSelect: () => ReactWrapper<P, S, C>): Promise<void>;
/**
 * Returns a string concatenating the labels for all selected options.
 */
export declare function getLabelText<P, S, C>(getSelect: () => ReactWrapper<P, S, C>): string;
/**
 * Removes the `option` by clicking its "X" button. Can only be used with a <Select/>
 * component with `mode="multiple"`. The provided strings must match the option label
 * exactly.
 */
export declare function removeMultiSelectOption<P, S, C>(getSelect: () => ReactWrapper<P, S, C>, option: string): void;
/**
 * Selects options from the dropdown menu for a <Select/> component with `mode="multiple"`.
 * The provided strings must match the option labels exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export declare function multiSelect<P, S, C>(getSelect: () => ReactWrapper<P, S, C>, options: (string | RegExp)[]): Promise<void>;
/**
 * Selects options from the dropdown menu for a <Select/> component without a
 * mode. The provided string must match an option label exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
export declare function singleSelect<P, S, C>(getSelect: () => ReactWrapper<P, S, C>, option: string | RegExp): Promise<void>;
/**
 * Opens the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already opened or if the menu is unable to be opened.
 */
export declare function openMenu<P, S, C>(getSelect: () => ReactWrapper<P, S, C>): Promise<void>;
/**
 * Opens the dropdown menu, finds all of the options in the dropdown, closes
 * the menu, and returns a list of the text of each option in order.
 */
export declare function getAllOptions<P, S, C>(getSelect: () => ReactWrapper<P, S, C>): Promise<string[]>;
/**
 * Creates a new option for a Select with `mode="tags"` by typing it into the input,
 * clicking on the option in the options list, and then closing the menu.
 */
export declare function createNewOption<P, S, C>(getSelect: () => ReactWrapper<P, S, C>, option: string): Promise<void>;
//# sourceMappingURL=selectEvent.d.ts.map