import type { CSSProperties } from 'react';
import type { WithLoadingState } from '../LoadingState/LoadingState';
interface GenericSkeletonProps extends WithLoadingState {
    /** Label for screen readers */
    label?: React.ReactNode;
    /** fps for animation. Default is 60 fps. A lower number will use less resources. */
    frameRate?: number;
    /** Style property */
    style?: CSSProperties;
    /** Class name property */
    className?: string;
}
export declare const GenericSkeleton: ({ label, frameRate, style, loading, loadingDescription, ...restProps }: GenericSkeletonProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=GenericSkeleton.d.ts.map