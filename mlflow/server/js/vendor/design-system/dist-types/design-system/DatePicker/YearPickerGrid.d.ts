import type { ReactNode } from 'react';
export interface YearPickerGridProps {
    /** The currently selected year as a Date. Only the year component is used. */
    value?: Date;
    /** Callback fired when a year is selected. The Date is set to Jan 1 of the selected year. */
    onSelect: (date: Date) => void;
    /** Optional: The minimum selectable date. Only the year component is used. */
    min?: Date;
    /** Optional: The maximum selectable date. Only the year component is used. */
    max?: Date;
    /** Optional: Start of a date range. Used to highlight cells between rangeFrom and rangeTo. */
    rangeFrom?: Date;
    /** Optional: End of a date range. Used to highlight cells between rangeFrom and rangeTo. */
    rangeTo?: Date;
    /** Optional: Receives the cell date and the full default button element. The caller can wrap it
     * (e.g., add a dot indicator) or replace it entirely (e.g., to apply custom `disabled` logic). */
    renderCellContent?: (date: Date, defaultContent: ReactNode) => ReactNode;
    /** Optional: Additional predicate to disable individual year cells beyond `min`/`max`. */
    isDateDisabled?: (date: Date) => boolean;
}
/**
 * YearPickerGrid renders a 4x3 grid of years with decade navigation.
 * It is used inside the DatePicker when `granularity='year'` is set.
 */
export declare function YearPickerGrid({ value, onSelect, min, max, rangeFrom, rangeTo, renderCellContent, isDateDisabled }: YearPickerGridProps): JSX.Element;
//# sourceMappingURL=YearPickerGrid.d.ts.map