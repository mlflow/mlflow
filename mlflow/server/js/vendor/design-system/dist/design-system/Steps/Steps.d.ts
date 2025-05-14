import type { StepsProps as AntDStepsProps } from 'antd';
import type { ReactNode } from 'react';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface StepsProps extends AntDStepsProps, DangerouslySetAntdProps<AntDStepsProps>, HTMLDataAttributes {
    children?: ReactNode;
}
/** @deprecated Please use the supported Stepper widget instead. See https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-stepper--docs */
export declare const Steps: {
    ({ dangerouslySetAntdProps, ...props }: StepsProps): JSX.Element;
    Step: import("react").ClassicComponentClass<import("antd").StepProps>;
};
//# sourceMappingURL=Steps.d.ts.map