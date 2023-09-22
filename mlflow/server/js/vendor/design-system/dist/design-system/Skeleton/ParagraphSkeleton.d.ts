import type { CSSProperties } from 'react';
import type { WithLoadingState } from '../LoadingState/LoadingState';
interface ParagraphSkeletonProps extends WithLoadingState {
    /** Label for screen readers */
    label?: React.ReactNode;
    /** Seed that deterministically arranges the uneven lines, so that they look like ragged text.
     * If you don't provide this (or give each skeleton the same seed) they will all look the same. */
    seed?: string;
    /** fps for animation. Default is 60 fps. A lower number will use less resources. */
    frameRate?: number;
    /** Style property */
    style?: CSSProperties;
    /** Class name property */
    className?: string;
}
export declare const ParagraphSkeleton: ({ label, seed, frameRate, style, loading, loadingDescription, ...restProps }: ParagraphSkeletonProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=ParagraphSkeleton.d.ts.map