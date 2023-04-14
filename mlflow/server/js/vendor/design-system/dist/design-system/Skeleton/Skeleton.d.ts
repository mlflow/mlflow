/// <reference types="react" />
import type { SkeletonProps as AntDSkeletonProps } from 'antd';
import { Skeleton as AntDSkeleton } from 'antd';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface SkeletonProps extends AntDSkeletonProps, DangerouslySetAntdProps<AntDSkeletonProps>, HTMLDataAttributes {
    label?: React.ReactNode;
}
interface SkeletonInterface extends React.FC<SkeletonProps> {
    Button: typeof AntDSkeleton.Button;
    Image: typeof AntDSkeleton.Image;
    Input: typeof AntDSkeleton.Input;
}
export declare const Skeleton: SkeletonInterface;
export {};
//# sourceMappingURL=Skeleton.d.ts.map