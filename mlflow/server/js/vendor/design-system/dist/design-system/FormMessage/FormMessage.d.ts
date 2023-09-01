/// <reference types="react" />
import type { Interpolation, Theme as EmotionTheme } from '@emotion/react';
import type { ValidationState } from '../types';
export interface FormMessageProps {
    message: React.ReactNode;
    type: ValidationState;
    className?: string;
    css?: Interpolation<EmotionTheme>;
}
export declare function FormMessage({ message, type, className, css }: FormMessageProps): JSX.Element;
//# sourceMappingURL=FormMessage.d.ts.map