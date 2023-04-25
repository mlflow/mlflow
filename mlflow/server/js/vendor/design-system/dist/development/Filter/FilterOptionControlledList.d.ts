/// <reference types="react" />
import type { FilterOptionListProps } from './FilterOptionList';
export interface FilterOptionControlledListProps extends Omit<FilterOptionListProps, 'children'> {
    withSearch?: boolean;
    showSelectAndClearAll?: boolean;
    options: string[];
}
export declare const FilterOptionControlledList: import("react").ForwardRefExoticComponent<FilterOptionControlledListProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=FilterOptionControlledList.d.ts.map