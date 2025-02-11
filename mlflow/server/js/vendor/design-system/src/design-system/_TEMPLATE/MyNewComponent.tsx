import type { ButtonProps as AntDButtonProps } from 'antd';
import { Button as AntDButton } from 'antd';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';

export interface MyNewComponentProps extends AntDButtonProps {
  /** These props all come from `antd`, and Ant doesn't annotate their
   * types (unfortunately). But, we can optionally decide to annotate them
   * on an individual basis. Pretty cool!
   */
  href?: AntDButtonProps['href'];
}

export const MyNewComponent = (props: MyNewComponentProps): JSX.Element => {
  return (
    <DesignSystemAntDConfigProvider>
      <AntDButton {...props} />
    </DesignSystemAntDConfigProvider>
  );
};

const defaultProps: Partial<MyNewComponentProps> = {
  type: 'primary',
};

MyNewComponent.defaultProps = defaultProps;
