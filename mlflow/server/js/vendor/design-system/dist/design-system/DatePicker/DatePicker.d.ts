/// <reference types="react" />
import type { DatePickerProps as AntDDatePickerProps, MonthPickerProps as AntDMonthPickerProps, RangePickerProps as AntDRangePickerProps } from 'antd/lib/date-picker';
export type DatePickerRef = {
    focus: () => void;
    blur: () => void;
};
export type DatePickerProps = AntDDatePickerProps & React.RefAttributes<DatePickerRef>;
export type RangePickerProps = AntDRangePickerProps & React.RefAttributes<DatePickerRef>;
export type TimePickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef>;
export type QuarterPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef>;
export type WeekPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef>;
export type MonthPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef>;
export type YearPickerProps = AntDMonthPickerProps & React.RefAttributes<DatePickerRef>;
export declare const DuboisDatePicker: React.VFC<DatePickerProps>;
/**
 * `LegacyDatePicker` was added as a temporary solution pending an
 * official Du Bois replacement. Use with caution.
 * @deprecated
 */
export declare const LegacyDatePicker: import("react").VFC<DatePickerProps> & {
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    RangePicker: import("react").VFC<RangePickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    TimePicker: import("react").VFC<TimePickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    QuarterPicker: import("react").VFC<QuarterPickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    WeekPicker: import("react").VFC<WeekPickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    MonthPicker: import("react").VFC<MonthPickerProps>;
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    YearPicker: import("react").VFC<YearPickerProps>;
};
//# sourceMappingURL=DatePicker.d.ts.map