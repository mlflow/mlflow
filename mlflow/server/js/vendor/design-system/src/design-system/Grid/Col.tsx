import type { ColProps as AntDColProps } from 'antd';
import { Col as AntDCol } from 'antd';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';

export interface ColProps extends AntDColProps, DangerouslySetAntdProps<AntDColProps>, HTMLDataAttributes {}

export const Col: React.FC<ColProps> = ({ dangerouslySetAntdProps, ...props }: ColProps) => (
  <DesignSystemAntDConfigProvider>
    <AntDCol {...props} {...dangerouslySetAntdProps} />
  </DesignSystemAntDConfigProvider>
);
