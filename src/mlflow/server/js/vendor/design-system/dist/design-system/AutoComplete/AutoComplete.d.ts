/// <reference types="react" />
import type { AutoCompleteProps as AntDAutoCompleteProps } from 'antd';
import type { OptionType } from 'antd/es/select';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface AutoCompleteProps extends AntDAutoCompleteProps {
}
export interface AutoCompleteProps extends AntDAutoCompleteProps, DangerouslySetAntdProps<AntDAutoCompleteProps>, HTMLDataAttributes {
}
interface AutoCompleteInterface extends React.FC<AutoCompleteProps> {
    Option: OptionType;
}
/**
 * @deprecated Use `TypeaheadCombobox` instead.
 */
export declare const AutoComplete: AutoCompleteInterface;
export {};
//# sourceMappingURL=AutoComplete.d.ts.map