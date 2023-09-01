/// <reference types="react" />
import type { Input as AntDInput } from 'antd';
import type { InputProps } from '../Input';
import type { HTMLDataAttributes } from '../types';
interface TableFilterInputProps extends InputProps, HTMLDataAttributes {
    onSubmit?: () => void;
    showSearchButton?: boolean;
}
export declare const TableFilterInput: import("react").ForwardRefExoticComponent<TableFilterInputProps & import("react").RefAttributes<AntDInput>>;
export {};
//# sourceMappingURL=TableFilterInput.d.ts.map