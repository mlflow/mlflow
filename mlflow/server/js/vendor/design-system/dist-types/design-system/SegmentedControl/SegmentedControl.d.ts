import type { RadioGroupProps } from 'antd';
import type { RadioButtonProps } from 'antd/lib/radio/radioButton';
import React from 'react';
import type { ButtonSize } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface SegmentedControlButtonProps extends Omit<RadioButtonProps, 'optionType' | 'buttonStyle' | 'prefixCls' | 'skipGroup' | 'title'>, DangerouslySetAntdProps<RadioButtonProps>, HTMLDataAttributes {
    icon?: React.ReactNode;
    /**
     * Title text to display when hovering over the button using the browser's native tooltip.
     */
    title?: string;
}
export declare const SegmentedControlButton: React.ForwardRefExoticComponent<SegmentedControlButtonProps & React.RefAttributes<HTMLButtonElement>>;
export interface SegmentedControlGroupProps extends Omit<RadioGroupProps, 'size'>, DangerouslySetAntdProps<RadioGroupProps>, HTMLDataAttributes, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange | DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    size?: ButtonSize;
    spaced?: boolean;
    name: string;
    newStyleFlagOverride?: boolean;
    'aria-labelledby'?: string;
}
export declare const SegmentedControlGroup: React.ForwardRefExoticComponent<SegmentedControlGroupProps & React.RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=SegmentedControl.d.ts.map