import type { ForwardedRef, HTMLAttributes, ReactNode } from 'react';
import type { DayPickerRangeProps, DayPickerSingleProps } from 'react-day-picker';
import type { InputProps } from '../Input/common';
import type { HTMLDataAttributes, ValidationState } from '../types';
/**
 * The granularity level for the DatePicker.
 * - `'day'` (default): Standard day calendar picker
 * - `'month'`: 4x3 grid of month names with year navigation
 * - `'year'`: 4x3 grid of years with decade navigation
 * - `'week'`: Day calendar with ISO week selection (clicking a day selects the entire week)
 */
export type DatePickerGranularity = 'day' | 'month' | 'year' | 'week';
export interface DatePickerChangeEventType {
    target: {
        name?: string;
        value: Date | undefined;
        textValue?: string;
    };
    type: string;
    updateLocation: 'input' | 'calendar';
}
export interface DateRange {
    from: Date | undefined;
    to: Date | undefined;
}
export interface RangePickerChangeEventType {
    target: {
        name?: string;
        value: DateRange | undefined;
        rangeInString?: {
            from?: string;
            to?: string;
        };
    };
    type: string;
    updateLocation: 'input' | 'calendar' | 'preset';
}
export interface DatePickerWrapperProps {
    wrapperProps?: HTMLAttributes<HTMLDivElement> & HTMLDataAttributes;
}
export interface DatePickerProps extends Omit<InputProps, 'type' | 'suffix' | 'onKeyDown' | 'value' | 'onChange' | 'max' | 'min' | 'size'>, DatePickerWrapperProps {
    onChange?: (e: DatePickerChangeEventType) => void;
    onClear?: () => void;
    open?: boolean;
    onOpenChange?: (visible: boolean) => void;
    value?: Date | string;
    validationState?: ValidationState;
    includeTime?: boolean;
    includeSeconds?: boolean;
    /**
     * Expected format HH:mm or HH:mm:ss
     */
    defaultTime?: string;
    datePickerProps?: Omit<DayPickerSingleProps, 'mode' | 'selected'> | Omit<DayPickerRangeProps, 'mode' | 'selected'>;
    timeInputProps?: Omit<InputProps, 'type' | 'allowClear' | 'onChange' | 'value' | 'componentId' | 'size'>;
    name?: string;
    width?: string | number;
    maxWidth?: string | number;
    minWidth?: string | number;
    /** Predicate to disable individual calendar cells across all granularities (day/week/month/year). */
    dateTimeDisabledFn?: (date: Date) => boolean;
    quickActions?: DatePickerQuickActionProps[];
    onOkPress?: () => void;
    okButtonLabel?: string;
    min?: Date | string | number | undefined;
    max?: Date | string | number | undefined;
    /**
     * DO NOT USE THIS PROP. This is only for internal use.
     */
    showTimeZone?: boolean;
    /**
     * Custom timezone label, this has no functional impact, converting to the correct timezone must be done outside this component
     */
    customTimeZoneLabel?: string;
    /**
     * When enabled, changes input type to 'text' and allows users to type custom date strings directly (e.g., "now-2d", "now-1d")
     * The raw string value will be passed through onChange, allowing the consumer to parse it
     */
    allowCustomInput?: boolean;
    onCustomInputChange?: (value: string) => void;
    /**
     * Controls which calendar panel UI is shown.
     * - `'day'` (default): Standard day calendar
     * - `'month'`: 4x3 month grid with year navigation
     * - `'year'`: 4x3 year grid with decade navigation
     * - `'week'`: Day calendar with ISO week row selection
     */
    granularity?: DatePickerGranularity;
    /**
     * Renders content inside each calendar cell. Receives the cell date and the default cell content
     * the full default button element for that cell, so consumers can wrap it (e.g., add a dot
     * indicator) or replace it entirely (e.g., to apply custom `disabled` logic).
     * Applied to all granularities:
     * - `'day'`/`'week'`: `defaultContent` is the rendered day button (disabled state reflects
     *   `datePickerProps.disabled` and `dateTimeDisabledFn`). The button's ref is preserved, so
     *   keyboard navigation works when wrapping. Replacing the button loses focus management.
     * - `'month'`/`'year'`: `defaultContent` is the rendered month/year button (disabled state
     *   reflects `min`/`max` and `dateTimeDisabledFn`).
     */
    renderCellContent?: (date: Date, defaultContent: ReactNode) => ReactNode;
    /**
     * Content rendered below the calendar grid inside the popover (e.g., a loading spinner).
     * Mirrors the legacy `renderExtraFooter` prop.
     */
    footer?: ReactNode;
}
interface DatePickerQuickActionProps {
    label: string;
    /**
     * Do not pass Date[] as value, it's only for internal use
     */
    value: Date | Date[];
    onClick?: (value: Date | Date[]) => void;
}
export declare const getDatePickerQuickActionBasic: ({ today, yesterday, sevenDaysAgo, }: {
    today?: Partial<DatePickerQuickActionProps> | undefined;
    yesterday?: Partial<DatePickerQuickActionProps> | undefined;
    sevenDaysAgo?: Partial<DatePickerQuickActionProps> | undefined;
}) => DatePickerQuickActionProps[];
export declare const DatePicker: import("react").ForwardRefExoticComponent<DatePickerProps & import("react").RefAttributes<HTMLInputElement>>;
export interface RangePickerProps extends Omit<DayPickerRangeProps, 'mode'>, DatePickerWrapperProps {
    name?: string;
    onChange?: (e: RangePickerChangeEventType) => void;
    startDatePickerProps?: DatePickerProps & {
        ref?: ForwardedRef<HTMLInputElement>;
    };
    endDatePickerProps?: DatePickerProps & {
        ref?: ForwardedRef<HTMLInputElement>;
    };
    includeTime?: boolean;
    includeSeconds?: boolean;
    allowClear?: boolean;
    /**
     * Minimum recommended width 300px without `includeTime` and 350px with `includeTime`. 400px if both `includeTime` and `allowClear` are true
     */
    width?: string | number;
    /**
     * Minimum recommended width 300px without `includeTime` and 350px with `includeTime`. 400px if both `includeTime` and `allowClear` are true
     */
    maxWidth?: string | number;
    /**
     * Minimum recommended width 300px without `includeTime` and 350px with `includeTime`. 400px if both `includeTime` and `allowClear` are true
     */
    minWidth?: string | number;
    disabled?: boolean;
    quickActions?: RangePickerQuickActionProps[];
    /**
     * Allow the user to select a range that has a larger start date than end date or a start date that is after the end date
     */
    noRangeValidation?: boolean;
    /**
     * When enabled, changes input type to 'text' and allows users to type custom date strings directly (e.g., "now-2d", "now-1d")
     * The raw string value will be passed through onChange, allowing the consumer to parse it
     */
    allowCustomInput?: boolean;
    /**
     * Controls which calendar panel UI is shown for both start and end date pickers.
     * @see DatePickerGranularity
     */
    granularity?: DatePickerGranularity;
}
export interface DateRangeValue {
    from: Date | string | undefined;
    to: Date | string | undefined;
}
export interface RangePickerQuickActionProps extends DatePickerQuickActionProps {
}
export declare const getRangeQuickActionsBasic: ({ today, yesterday, lastWeek, }: {
    today?: Partial<RangePickerQuickActionProps> | undefined;
    yesterday?: Partial<RangePickerQuickActionProps> | undefined;
    lastWeek?: Partial<RangePickerQuickActionProps> | undefined;
}) => RangePickerQuickActionProps[];
export declare const RangePicker: (props: RangePickerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=DatePicker.d.ts.map