import type { CSSProperties } from 'react';
interface TitleSkeletonProps {
    /** Label for screen readers */
    label?: React.ReactNode;
    /** fps for animation. Default is 60 fps. A lower number will use less resources. */
    frameRate?: number;
    /** Style property */
    style?: CSSProperties;
}
export declare const TitleSkeleton: ({ label, frameRate, style }: TitleSkeletonProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=TitleSkeleton.d.ts.map