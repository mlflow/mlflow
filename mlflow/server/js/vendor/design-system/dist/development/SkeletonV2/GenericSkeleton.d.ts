import type { CSSProperties } from 'react';
interface GenericSkeletonProps {
    /** Label for screen readers */
    label?: React.ReactNode;
    /** fps for animation. Default is 60 fps. A lower number will use less resources. */
    frameRate?: number;
    /** Style property */
    style?: CSSProperties;
}
export declare const GenericSkeleton: ({ label, frameRate, style }: GenericSkeletonProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=GenericSkeleton.d.ts.map