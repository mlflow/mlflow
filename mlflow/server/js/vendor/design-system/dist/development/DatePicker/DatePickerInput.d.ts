import { type InputHTMLAttributes } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system';
import type { AnalyticsEventProps, ValidationState } from '../../design-system/types';
export interface DatePickerInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'prefix' | 'suffix'>, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange | DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    showTimeZone?: boolean;
    validationState?: ValidationState;
    prefix?: React.ReactNode;
    suffix?: React.ReactNode;
    allowClear?: boolean;
    onClear?: () => void;
}
export declare const DatePickerInput: import("react").ForwardRefExoticComponent<DatePickerInputProps & import("react").RefAttributes<HTMLInputElement>>;
//# sourceMappingURL=DatePickerInput.d.ts.map