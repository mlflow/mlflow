import type { UseComboboxReturnValue } from 'downshift';
import React from 'react';
import type { HTMLAttributes, ReactNode } from 'react';
export interface TypeaheadComboboxMenuProps<T> extends HTMLAttributes<HTMLUListElement> {
    comboboxState: UseComboboxReturnValue<T>;
    loading?: boolean;
    emptyText?: string | React.ReactNode;
    width?: number | string;
    minWidth?: number;
    maxWidth?: number | string;
    minHeight?: number;
    maxHeight?: number;
    listWrapperHeight?: number;
    virtualizerRef?: React.RefObject<T>;
    children?: ReactNode;
    matchTriggerWidth?: boolean;
}
export declare const TypeaheadComboboxMenu: React.ForwardRefExoticComponent<TypeaheadComboboxMenuProps<any> & React.RefAttributes<HTMLElement | null>>;
//# sourceMappingURL=TypeaheadComboboxMenu.d.ts.map