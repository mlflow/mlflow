import type { Input as AntDInput } from 'antd';
import React from 'react';
import type { InputProps } from '../Input';
import type { HTMLDataAttributes } from '../types';
interface TableFilterInputProps extends InputProps, HTMLDataAttributes {
    onSubmit?: () => void;
    onClear?: () => void;
    showSearchButton?: boolean;
}
export declare const TableFilterInput: React.ForwardRefExoticComponent<TableFilterInputProps & React.RefAttributes<AntDInput>>;
export {};
//# sourceMappingURL=TableFilterInput.d.ts.map