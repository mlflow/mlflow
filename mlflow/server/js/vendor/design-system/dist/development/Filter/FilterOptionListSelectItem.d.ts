/// <reference types="react" />
import type { HTMLDataAttributes } from '../../design-system/types';
export interface FilterOptionListSelectItemProps extends HTMLDataAttributes {
    value: string;
    checked?: boolean;
    children?: React.ReactNode;
    onChange?: (...args: any[]) => any;
}
export declare const FilterOptionListSelectItem: import("react").ForwardRefExoticComponent<FilterOptionListSelectItemProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=FilterOptionListSelectItem.d.ts.map