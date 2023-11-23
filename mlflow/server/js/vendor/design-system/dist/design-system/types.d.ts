import type { Interpolation } from '@emotion/react';
import type { Theme } from '../theme';
import type { ColorVars } from './constants';
export interface DangerouslySetAntdProps<P> {
    /** For components that wrap `antd` components, emergency access for properties we do not support. Ask in #dubois before using. */
    dangerouslySetAntdProps?: P;
}
export interface DangerousGeneralProps {
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<Theme>;
}
/** Mapping for color variables can be found under constants.tsx */
export type TypographyColor = keyof typeof ColorVars;
/** Generic type for supporting data- attributes */
export interface HTMLDataAttributes {
    [key: `data-${string}`]: string;
}
export type ValidationState = 'success' | 'warning' | 'error';
export interface FormElementValidationState {
    validationState?: ValidationState;
}
//# sourceMappingURL=types.d.ts.map