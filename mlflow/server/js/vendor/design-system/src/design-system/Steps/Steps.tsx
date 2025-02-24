import type { StepsProps as AntDStepsProps } from 'antd';
import { Steps as AntDSteps } from 'antd';
import type { ReactNode } from 'react';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface StepsProps extends AntDStepsProps, DangerouslySetAntdProps<AntDStepsProps>, HTMLDataAttributes {
  children?: ReactNode;
}

export const Steps = /* #__PURE__ */ (() => {
  function Steps({ dangerouslySetAntdProps, ...props }: StepsProps): JSX.Element {
    return (
      <DesignSystemAntDConfigProvider>
        <AntDSteps {...addDebugOutlineIfEnabled()} {...props} {...dangerouslySetAntdProps} />
      </DesignSystemAntDConfigProvider>
    );
  }

  Steps.Step = AntDSteps.Step;

  return Steps;
})();
