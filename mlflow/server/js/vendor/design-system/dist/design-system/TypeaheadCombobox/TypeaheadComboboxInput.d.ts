import type { UseComboboxReturnValue } from 'downshift';
import type { InputProps } from '../Input';
export interface TypeaheadComboboxInputProps<T> extends Omit<InputProps, 'componentId' | 'analyticsEvents'> {
    comboboxState: UseComboboxReturnValue<T>;
    allowClear?: boolean;
    showComboboxToggleButton?: boolean;
    formOnChange?: (value: T) => void;
    clearInputValueOnFocus?: boolean;
}
export declare const TypeaheadComboboxInput: import("react").ForwardRefExoticComponent<Omit<TypeaheadComboboxInputProps<any>, "componentId" | "analyticsEvents"> & import("react").RefAttributes<HTMLDivElement | null>>;
//# sourceMappingURL=TypeaheadComboboxInput.d.ts.map