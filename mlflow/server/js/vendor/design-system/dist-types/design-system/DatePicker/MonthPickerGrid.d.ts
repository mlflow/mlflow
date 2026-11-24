import type { ReactNode } from 'react';
export interface MonthPickerGridProps {
    /** The currently selected month as a Date. Year and month components are used. */
    value?: Date;
    /** Callback fired when a month is selected. The Date is set to the 1st of the selected month. */
    onSelect: (date: Date) => void;
    /** Optional: The minimum selectable date. Year and month components are used. */
    min?: Date;
    /** Optional: The maximum selectable date. Year and month components are used. */
    max?: Date;
    /** Optional: Start of a date range. Used to highlight cells between rangeFrom and rangeTo. */
    rangeFrom?: Date;
    /** Optional: End of a date range. Used to highlight cells between rangeFrom and rangeTo. */
    rangeTo?: Date;
    /** Optional: Receives the cell date and the full default button element. The caller can wrap it
     * (e.g., add a dot indicator) or replace it entirely (e.g., to apply custom `disabled` logic). */
    renderCellContent?: (date: Date, defaultContent: ReactNode) => ReactNode;
    /** Optional: Additional predicate to disable individual month cells beyond `min`/`max`. */
    isDateDisabled?: (date: Date) => boolean;
}
/**
 * MonthPickerGrid renders a 4x3 grid of month names for a given year with year navigation.
 * It is used inside the DatePicker when `granularity='month'` is set.
 */
export declare function MonthPickerGrid({ value, onSelect, min, max, rangeFrom, rangeTo, renderCellContent, isDateDisabled }: MonthPickerGridProps): JSX.Element;
//# sourceMappingURL=MonthPickerGrid.d.ts.map