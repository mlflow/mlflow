/// <reference types="react" />
import type { UseComboboxReturnValue } from 'downshift';
import type { InputProps } from '../Input';
export interface TypeaheadComboboxInputProps<T> extends InputProps {
    comboboxState: UseComboboxReturnValue<T>;
    allowClear?: boolean;
}
export declare const TypeaheadComboboxInput: import("react").ForwardRefExoticComponent<TypeaheadComboboxInputProps<any> & import("react").RefAttributes<HTMLDivElement | null>>;
//# sourceMappingURL=TypeaheadComboboxInput.d.ts.map