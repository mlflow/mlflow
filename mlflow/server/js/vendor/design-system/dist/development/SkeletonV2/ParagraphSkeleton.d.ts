import type { CSSProperties } from 'react';
interface ParagraphSkeletonProps {
    /** Label for screen readers */
    label?: React.ReactNode;
    /** Seed that deterministically arranges the uneven lines, so that they look like ragged text.
     * If you don't provide this (or give each skeleton the same seed) they will all look the same. */
    seed?: string;
    /** fps for animation. Default is 60 fps. A lower number will use less resources. */
    frameRate?: number;
    /** Style property */
    style?: CSSProperties;
}
export declare const ParagraphSkeleton: ({ label, seed, frameRate, style }: ParagraphSkeletonProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=ParagraphSkeleton.d.ts.map