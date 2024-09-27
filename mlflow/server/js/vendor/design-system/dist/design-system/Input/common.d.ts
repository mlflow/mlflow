import type { Input as AntDInput, InputProps as AntDInputProps } from 'antd';
import type { GroupProps as AntDGroupProps, PasswordProps as AntDPasswordProps, TextAreaProps as AntDTextAreaProps } from 'antd/lib/input';
import type { TextAreaRef } from 'antd/lib/input/TextArea';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventProps, DangerousGeneralProps, DangerouslySetAntdProps, FormElementValidationState, HTMLDataAttributes } from '../types';
export type InputRef = AntDInput;
export type { TextAreaRef };
export interface InputProps extends Omit<AntDInputProps, 'prefixCls' | 'size' | 'addonAfter' | 'bordered'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDInputProps>, DangerousGeneralProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    onClear?: () => void;
}
export interface PasswordProps extends Omit<AntDPasswordProps, 'inputPrefixCls' | 'action' | 'visibilityToggle' | 'iconRender'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDPasswordProps>, DangerousGeneralProps {
}
export interface TextAreaProps extends Omit<AntDTextAreaProps, 'bordered' | 'showCount' | 'size'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDTextAreaProps>, DangerousGeneralProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
}
export interface InputGroupProps extends Omit<AntDGroupProps, 'size'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDGroupProps>, DangerousGeneralProps {
}
//# sourceMappingURL=common.d.ts.map