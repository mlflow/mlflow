import type { CSSObject, SerializedStyles } from '@emotion/react';
import { Input as AntDInput } from 'antd';
import React from 'react';
import type { InputProps } from './common';
import type { Theme } from '../../theme';
import type { ButtonSize } from '../Button';
import type { ValidationState } from '../types';
interface InputStylesOptions {
    useFocusWithin?: boolean;
}
export declare const getInputStyles: (clsPrefix: string, theme: Theme, { validationState, type, hasValue, useNewShadows, useNewFormUISpacing, useNewBorderRadii, locked, size, }: {
    validationState?: ValidationState;
    type?: string;
    hasValue?: boolean;
    useNewShadows?: boolean;
    useNewFormUISpacing?: boolean;
    useNewBorderRadii?: boolean;
    locked?: boolean;
    size?: ButtonSize;
}, { useFocusWithin }: InputStylesOptions) => CSSObject;
export declare const getInputEmotionStyles: (clsPrefix: string, theme: Theme, { validationState, type, hasValue, useNewShadows, useNewBorderRadii, locked, size, }: {
    validationState?: ValidationState;
    type?: string;
    hasValue?: boolean;
    useNewShadows: boolean;
    useNewBorderRadii?: boolean;
    locked?: boolean;
    size?: ButtonSize;
}) => SerializedStyles;
export declare const Input: React.ForwardRefExoticComponent<InputProps & React.RefAttributes<AntDInput>>;
export {};
//# sourceMappingURL=Input.d.ts.map