import type { SerializedStyles } from '@emotion/react';
import { Input as AntDInput } from 'antd';
import React from 'react';
import type { Theme } from '../../theme';
import type { ValidationState } from '../types';
import type { InputProps } from './common';
export declare const getInputEmotionStyles: (clsPrefix: string, theme: Theme, { validationState }: {
    validationState?: ValidationState | undefined;
}, useTransparent?: boolean) => SerializedStyles;
export declare const Input: React.ForwardRefExoticComponent<InputProps & React.RefAttributes<AntDInput>>;
//# sourceMappingURL=Input.d.ts.map