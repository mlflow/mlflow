/**
 * Extracts the display label from a combobox — equivalent to the selected
 * option's label.
 */
export declare function getDisplayLabel(combobox: HTMLElement): string;
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
export declare function getListbox(combobox: HTMLElement): HTMLElement;
/**
 * Returns all options associated with a combobox (requires the select to have
 * been opened).
 */
export declare function getOptions(combobox: HTMLElement): HTMLElement[];
/**
 * Returns the accessible name for each option in a combobox.
 */
export declare function getOptionNames(combobox: HTMLElement): string[];
//# sourceMappingURL=select.d.ts.map