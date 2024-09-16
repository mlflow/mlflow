import type { CSSObject, SerializedStyles } from '@emotion/react';
import { Input as AntDInput } from 'antd';
import React from 'react';
import type { InputProps } from './common';
import type { Theme } from '../../theme';
import type { ValidationState } from '../types';
interface InputStylesOptions {
    useTransparent?: boolean;
    useFocusWithin?: boolean;
}
export declare const getInputStyles: (clsPrefix: string, theme: Theme, { validationState, type, hasValue }: {
    validationState?: ValidationState;
    type?: string;
    hasValue?: boolean;
}, { useTransparent, useFocusWithin }: InputStylesOptions) => CSSObject;
export declare const getInputEmotionStyles: (clsPrefix: string, theme: Theme, { validationState, type, hasValue }: {
    validationState?: ValidationState;
    type?: string;
    hasValue?: boolean;
}, useTransparent?: boolean) => SerializedStyles;
export declare const Input: React.ForwardRefExoticComponent<InputProps & React.RefAttributes<AntDInput>>;
export {};
//# sourceMappingURL=Input.d.ts.map