import type { SkeletonProps as AntDSkeletonProps } from 'antd';
import { Skeleton as AntDSkeleton } from 'antd';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface LegacySkeletonProps extends WithLoadingState, AntDSkeletonProps, DangerouslySetAntdProps<AntDSkeletonProps>, HTMLDataAttributes {
    label?: React.ReactNode;
}
interface LegacySkeletonInterface extends React.FC<LegacySkeletonProps> {
    Button: typeof AntDSkeleton.Button;
    Image: typeof AntDSkeleton.Image;
    Input: typeof AntDSkeleton.Input;
}
/** @deprecated This component is deprecated. Use ParagraphSkeleton, TitleSkeleton, or GenericSkeleton instead. */
export declare const LegacySkeleton: LegacySkeletonInterface;
export {};
//# sourceMappingURL=LegacySkeleton.d.ts.map