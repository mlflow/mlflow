/// <reference types="react" />
import type { HTMLDataAttributes } from '../types';
export interface SpinnerProps extends HTMLDataAttributes {
    size?: 'small' | 'default' | 'large';
    className?: string;
    delay?: number;
    frameRate?: number;
}
export declare const Spinner: React.FC<SpinnerProps>;
//# sourceMappingURL=Spinner.d.ts.map