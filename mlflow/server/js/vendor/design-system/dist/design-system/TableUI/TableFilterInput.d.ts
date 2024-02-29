/// <reference types="react" />
import type { Input as AntDInput } from 'antd';
import { Button } from '../Button';
import type { InputProps } from '../Input';
import type { HTMLDataAttributes } from '../types';
interface TableFilterInputProps extends InputProps, HTMLDataAttributes {
    onSubmit?: () => void;
    showSearchButton?: boolean;
    searchButtonProps?: Omit<React.ComponentProps<typeof Button>, 'children' | 'type' | 'size' | 'componentId' | 'analyticsEvents'>;
    containerProps?: React.ComponentProps<'div'>;
}
export declare const TableFilterInput: import("react").ForwardRefExoticComponent<TableFilterInputProps & import("react").RefAttributes<AntDInput>>;
export {};
//# sourceMappingURL=TableFilterInput.d.ts.map