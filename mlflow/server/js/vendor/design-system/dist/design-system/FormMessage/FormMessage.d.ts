import type { Interpolation, Theme as EmotionTheme } from '@emotion/react';
import type { ValidationState } from '../types';
export interface FormMessageProps {
    id?: string;
    message: React.ReactNode;
    type: ValidationState;
    className?: string;
    css?: Interpolation<EmotionTheme>;
}
export declare function FormMessage({ id, message, type, className, css }: FormMessageProps): JSX.Element;
//# sourceMappingURL=FormMessage.d.ts.map