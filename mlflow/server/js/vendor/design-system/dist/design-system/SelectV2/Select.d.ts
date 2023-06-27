import type { DialogComboboxProps } from '../DialogCombobox';
export interface SelectV2Props extends Omit<DialogComboboxProps, 'multiselect' | 'value'> {
    placeholder?: string;
    value?: string;
}
export declare const SelectV2: (props: SelectV2Props) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Select.d.ts.map