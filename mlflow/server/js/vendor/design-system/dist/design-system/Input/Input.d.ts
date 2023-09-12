import type { InputProps as AntDInputProps } from 'antd';
import { Input as AntDInput } from 'antd';
import type { GroupProps as AntDGroupProps, PasswordProps as AntDPasswordProps, TextAreaProps as AntDTextAreaProps } from 'antd/lib/input';
import type { TextAreaRef } from 'antd/lib/input/TextArea';
import React from 'react';
import type { DangerousGeneralProps, DangerouslySetAntdProps, FormElementValidationState, HTMLDataAttributes } from '../types';
export type InputRef = AntDInput;
export type { TextAreaRef };
export interface InputProps extends Omit<AntDInputProps, 'prefixCls' | 'size' | 'addonAfter' | 'bordered'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDInputProps>, DangerousGeneralProps {
    onClear?: () => void;
}
export interface TextAreaProps extends Omit<AntDTextAreaProps, 'bordered' | 'showCount' | 'size'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDTextAreaProps>, DangerousGeneralProps {
}
export interface PasswordProps extends Omit<AntDPasswordProps, 'inputPrefixCls' | 'action' | 'visibilityToggle' | 'iconRender'>, FormElementValidationState, HTMLDataAttributes, DangerouslySetAntdProps<AntDPasswordProps>, DangerousGeneralProps {
}
export interface InputGroupProps extends Omit<AntDGroupProps, 'size'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDGroupProps>, DangerousGeneralProps {
}
export declare const Input: React.ForwardRefExoticComponent<InputProps & React.RefAttributes<AntDInput>> & {
    TextArea: React.ForwardRefExoticComponent<TextAreaProps & React.RefAttributes<TextAreaRef>>;
    Password: React.FC<PasswordProps>;
    Group: ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, compact, ...props }: InputGroupProps) => import("@emotion/react/jsx-runtime").JSX.Element;
};
export declare const __INTERNAL_DO_NOT_USE__TextArea: React.ForwardRefExoticComponent<TextAreaProps & React.RefAttributes<TextAreaRef>>;
export declare const __INTERNAL_DO_NOT_USE__Password: React.FC<PasswordProps>;
export declare const __INTERNAL_DO_NOT_USE_DEDUPE__Group: ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, compact, ...props }: InputGroupProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Input.d.ts.map