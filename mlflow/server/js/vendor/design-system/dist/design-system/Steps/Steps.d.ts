import type { StepsProps as AntDStepsProps } from 'antd';
import type { ReactNode } from 'react';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface StepsProps extends AntDStepsProps, DangerouslySetAntdProps<AntDStepsProps>, HTMLDataAttributes {
    children?: ReactNode;
}
export declare const Steps: {
    ({ dangerouslySetAntdProps, ...props }: StepsProps): JSX.Element;
    Step: import("react").ClassicComponentClass<import("antd").StepProps>;
};
//# sourceMappingURL=Steps.d.ts.map