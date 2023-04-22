/// <reference types="react" />
import type { HTMLDataAttributes } from '../../design-system/types';
export interface FilterOptionListCheckboxItemProps extends HTMLDataAttributes {
    value: string;
    checked?: boolean;
    indeterminate?: boolean;
    children?: React.ReactNode;
    onChange?: (...args: any[]) => any;
}
export declare const FilterOptionListCheckboxItem: import("react").ForwardRefExoticComponent<FilterOptionListCheckboxItemProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=FilterOptionListCheckboxItem.d.ts.map