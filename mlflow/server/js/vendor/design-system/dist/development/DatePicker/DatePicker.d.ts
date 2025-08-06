import type { ForwardedRef, HTMLAttributes } from 'react';
import type { DateRange as DayPickerDateRange, DayPickerRangeProps, DayPickerSingleProps } from 'react-day-picker';
import type { InputProps } from '../../design-system';
import type { HTMLDataAttributes, ValidationState } from '../../design-system/types';
export interface DatePickerChangeEventType {
    target: {
        name?: string;
        value: Date | undefined;
    };
    type: string;
    updateLocation: 'input' | 'calendar';
}
export interface RangePickerChangeEventType {
    target: {
        name?: string;
        value: DateRange | undefined;
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
    value?: Date;
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
    today?: Partial<DatePickerQuickActionProps>;
    yesterday?: Partial<DatePickerQuickActionProps>;
    sevenDaysAgo?: Partial<DatePickerQuickActionProps>;
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
}
export interface DateRange extends DayPickerDateRange {
}
export interface RangePickerQuickActionProps extends DatePickerQuickActionProps {
}
export declare const getRangeQuickActionsBasic: ({ today, yesterday, lastWeek, }: {
    today?: Partial<RangePickerQuickActionProps>;
    yesterday?: Partial<RangePickerQuickActionProps>;
    lastWeek?: Partial<RangePickerQuickActionProps>;
}) => RangePickerQuickActionProps[];
export declare const RangePicker: (props: RangePickerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=DatePicker.d.ts.map