import type { RowProps as AntDRowProps } from 'antd';
import { Row as AntDRow } from 'antd';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';

export interface RowProps extends AntDRowProps, DangerouslySetAntdProps<AntDRowProps>, HTMLDataAttributes {}

export const ROW_GUTTER_SIZE = 8;

export const Row: React.FC<RowProps> = ({ gutter = ROW_GUTTER_SIZE, ...props }) => {
  return (
    <DesignSystemAntDConfigProvider>
      <AntDRow gutter={gutter} {...props} />
    </DesignSystemAntDConfigProvider>
  );
};
