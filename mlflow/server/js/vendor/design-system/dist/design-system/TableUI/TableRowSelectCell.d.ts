/// <reference types="react" />
import type { HTMLDataAttributes } from '../types';
interface TableRowSelectCellProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
    /** Called when the checkbox is clicked */
    onChange?: (event: unknown) => void;
    /** Whether the checkbox is checked */
    checked?: boolean;
    /** Whether the row is indeterminate. Should only be used in header rows. */
    indeterminate?: boolean;
    /** Don't render a checkbox; used for providing spacing in header if you don't want "Select All" functionality */
    noCheckbox?: boolean;
    /** Whether the checkbox is disabled */
    isDisabled?: boolean;
}
export declare const TableRowSelectCell: import("react").ForwardRefExoticComponent<TableRowSelectCellProps & import("react").RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=TableRowSelectCell.d.ts.map