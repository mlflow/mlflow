import type { ReactNode } from 'react';
export interface ListboxOption {
    value: string;
    label: string;
    renderOption?: (additionalProps: any) => ReactNode;
    href?: string;
}
export interface ListboxProps {
    /**
     * Array of options to display in the listbox
     */
    options: ListboxOption[];
    /**
     * Callback fired when an option is selected
     */
    onSelect?: (value: string) => void;
    /**
     * Whether to include a filter input above the listbox
     * @default false
     */
    includeFilterInput?: boolean;
    /**
     * Placeholder text for the filter input
     * Only used when includeFilterInput is true
     */
    filterInputPlaceholder?: string;
    /**
     * Accessible label for the listbox
     */
    'aria-label': string;
    /**
     * Additional class name for the root element
     */
    className?: string;
    /**
     * Sets the initial selected value of the listbox
     */
    initialSelectedValue?: string;
    /**
     * Message to display if no options are found after filtering
     */
    filterInputEmptyMessage?: string;
}
export interface ListboxRootProps {
    children: ReactNode;
    className?: string;
    onSelect?: (value: string) => void;
    initialSelectedValue?: string;
    listBoxDivRef?: React.RefObject<HTMLDivElement> | null;
}
export interface ListboxInputProps {
    placeholder?: string;
    value: string;
    onChange: (value: string) => void;
    'aria-controls': string;
    'aria-activedescendant'?: string;
    className?: string;
    options: ListboxOption[];
}
export interface ListboxOptionsProps {
    options: ListboxOption[];
    selectedValue?: string;
    highlightedValue?: string;
    onSelect: (value: string) => void;
    onHighlight: (value: string) => void;
    className?: string;
}
export interface ListboxContextValue {
    selectedValue?: string;
    highlightedValue?: string;
    setHighlightedValue: (value: string) => void;
    setSelectedValue: (value: string) => void;
    listboxId: string;
    handleKeyNavigation: (event: React.KeyboardEvent, options: ListboxOption[]) => void;
}
//# sourceMappingURL=types.d.ts.map