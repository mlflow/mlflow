import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { HTMLDataAttributes } from '../types';
export interface SpinnerProps extends HTMLDataAttributes, WithLoadingState {
    size?: 'small' | 'default' | 'large';
    className?: string;
    delay?: number;
    frameRate?: number;
    label?: React.ReactNode;
    animationDuration?: number;
    inheritColor?: boolean;
}
export declare const Spinner: React.FC<SpinnerProps>;
//# sourceMappingURL=Spinner.d.ts.map