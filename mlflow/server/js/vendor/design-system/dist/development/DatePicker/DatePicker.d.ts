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
export interface DatePickerWrapperProps {
    wrapperProps?: HTMLAttributes<HTMLDivElement> & HTMLDataAttributes;
}
export interface DatePickerProps extends Omit<InputProps, 'type' | 'suffix' | 'onKeyDown' | 'value' | 'onChange'>, DatePickerWrapperProps {
    onChange?: (e: DatePickerChangeEventType) => void;
    onClear?: () => void;
    open?: boolean;
    onOpenChange?: (visible: boolean) => void;
    value?: Date;
    validationState?: ValidationState;
    includeTime?: boolean;
    /**
     * Expected format HH:mm
     */
    defaultTime?: string;
    datePickerProps?: Omit<DayPickerSingleProps, 'mode' | 'selected'> | Omit<DayPickerRangeProps, 'mode' | 'selected'>;
    timeInputProps?: Omit<InputProps, 'type' | 'allowClear' | 'onChange' | 'value'>;
    name?: string;
    width?: string | number;
    maxWidth?: string | number;
    minWidth?: string | number;
    dateTimeDisabledFn?: (date: Date) => boolean;
    quickActions?: DatePickerQuickActionProps[];
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
    onChange?: (date: DayPickerDateRange | undefined) => void;
    startDatePickerProps?: DatePickerProps & {
        ref?: ForwardedRef<HTMLInputElement>;
    };
    endDatePickerProps?: DatePickerProps & {
        ref?: ForwardedRef<HTMLInputElement>;
    };
    includeTime?: boolean;
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
}
export interface DateRange extends DayPickerDateRange {
}
interface RangePickerQuickActionProps extends DatePickerQuickActionProps {
}
export declare const getRangeQuickActionsBasic: ({ today, yesterday, lastWeek, }: {
    today?: Partial<RangePickerQuickActionProps>;
    yesterday?: Partial<RangePickerQuickActionProps>;
    lastWeek?: Partial<RangePickerQuickActionProps>;
}) => RangePickerQuickActionProps[];
export declare const RangePicker: (props: RangePickerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=DatePicker.d.ts.map