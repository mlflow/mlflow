/// <reference types="react" />
import type { HTMLDataAttributes } from '../../design-system/types';
export interface FilterOptionListProps extends HTMLDataAttributes {
    children: any;
    loading?: boolean;
    withProgressiveLoading?: boolean;
    onChange?: (...args: any[]) => any;
}
export declare const FilterOptionList: import("react").ForwardRefExoticComponent<FilterOptionListProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=FilterOptionList.d.ts.map