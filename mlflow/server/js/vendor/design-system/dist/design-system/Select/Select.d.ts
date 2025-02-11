import type { ConditionalOptionalLabel, DialogComboboxProps } from '../DialogCombobox';
export interface SelectProps extends Omit<DialogComboboxProps, 'multiselect' | 'value'> {
    placeholder?: string;
    value?: string;
    id?: string;
}
/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export declare const Select: (props: SelectProps & ConditionalOptionalLabel) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Select.d.ts.map