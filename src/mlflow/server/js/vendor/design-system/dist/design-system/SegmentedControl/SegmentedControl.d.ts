/// <reference types="react" />
import type { RadioGroupProps } from 'antd';
import type { RadioButtonProps } from 'antd/lib/radio/radioButton';
import type { ButtonSize } from '../Button';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface SegmentedControlButtonProps extends Omit<RadioButtonProps, 'optionType' | 'buttonStyle' | 'prefixCls' | 'skipGroup'>, DangerouslySetAntdProps<RadioButtonProps>, HTMLDataAttributes {
}
export declare const SegmentedControlButton: import("react").ForwardRefExoticComponent<SegmentedControlButtonProps & import("react").RefAttributes<HTMLButtonElement>>;
export interface SegmentedControlGroupProps extends Omit<RadioGroupProps, 'size'>, DangerouslySetAntdProps<RadioGroupProps>, HTMLDataAttributes {
    size?: ButtonSize;
    spaced?: boolean;
    name?: string;
}
export declare const SegmentedControlGroup: import("react").ForwardRefExoticComponent<SegmentedControlGroupProps & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=SegmentedControl.d.ts.map