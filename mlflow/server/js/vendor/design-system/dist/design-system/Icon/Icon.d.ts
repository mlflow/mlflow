import AntDIcon from '@ant-design/icons';
import type { ReactElement } from 'react';
import React from 'react';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
type AntDIconProps = Parameters<typeof AntDIcon>[0];
export type IconColors = 'danger' | 'warning' | 'success';
export interface IconProps extends Omit<AntDIconProps, 'component'>, DangerouslySetAntdProps<AntDIconProps>, HTMLDataAttributes, React.HTMLAttributes<HTMLSpanElement> {
    component?: (props: React.SVGProps<SVGSVGElement>) => ReactElement | null;
    color?: IconColors;
}
export declare const Icon: (props: IconProps) => JSX.Element;
export {};
//# sourceMappingURL=Icon.d.ts.map