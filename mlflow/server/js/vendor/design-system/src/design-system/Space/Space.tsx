import type { SpaceProps as AntDSpaceProps } from 'antd';
import { Space as AntDSpace } from 'antd';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';

export interface SpaceProps extends AntDSpaceProps, DangerouslySetAntdProps<AntDSpaceProps>, HTMLDataAttributes {}

export const Space: React.FC<SpaceProps> = ({ dangerouslySetAntdProps, ...props }) => {
  return (
    <DesignSystemAntDConfigProvider>
      <AntDSpace {...props} {...dangerouslySetAntdProps} />
    </DesignSystemAntDConfigProvider>
  );
};
