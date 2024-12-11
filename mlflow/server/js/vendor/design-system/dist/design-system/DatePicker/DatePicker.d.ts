import type { DatePickerProps as AntDDatePickerProps, MonthPickerProps as AntDMonthPickerProps, RangePickerProps as AntDRangePickerProps } from 'antd/lib/date-picker';
import React from 'react';
export type DatePickerRef = {
    focus: () => void;
    blur: () => void;
};
type DatePickerAccessibilityProps = {
    /**
     * ARIA live region attribute for the DatePicker component.
     * @default 'assertive'
     * It's recommended to use `assertive` as `polite` has been known to cause issues with screen readers with this specific component.
     */
    ariaLive?: 'assertive' | 'polite';
    wrapperDivProps?: React.HTMLAttributes<HTMLDivElement>;
};
export type DatePickerProps = AntDDatePickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type RangePickerProps = AntDRangePickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type TimePickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type QuarterPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type WeekPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type MonthPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export type YearPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef> & DatePickerAccessibilityProps;
export declare const DuboisDatePicker: React.VFC<DatePickerProps>;
/**
 * `LegacyDatePicker` was added as a temporary solution pending an
 * official Du Bois replacement. Use with caution.
 * @deprecated
 */
export declare const LegacyDatePicker: React.VFC<DatePickerProps> & {
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    RangePicker: React.VFC<RangePickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    TimePicker: React.VFC<TimePickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    QuarterPicker: React.VFC<QuarterPickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    WeekPicker: React.VFC<WeekPickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    MonthPicker: React.VFC<MonthPickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    YearPicker: React.VFC<YearPickerProps>;
};
export {};
//# sourceMappingURL=DatePicker.d.ts.map