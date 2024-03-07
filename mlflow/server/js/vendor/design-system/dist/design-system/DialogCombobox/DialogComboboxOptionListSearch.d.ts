import React from 'react';
import type { InputProps } from '../Input';
export interface DialogComboboxOptionListSearchProps extends InputProps {
    children: any;
    hasWrapper?: boolean;
    virtualized?: boolean;
    onSearch?: (value: string) => void;
    /** Set controlledValue and setControlledValue if search input is controlled, e.g. for custom filtering logic */
    controlledValue?: string;
    setControlledValue?: (value: string) => void;
}
export declare const DialogComboboxOptionListSearch: React.ForwardRefExoticComponent<DialogComboboxOptionListSearchProps & React.RefAttributes<import("antd/lib/input/Input").default>>;
//# sourceMappingURL=DialogComboboxOptionListSearch.d.ts.map