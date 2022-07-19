import { Cascader, CascaderProps } from 'antd';
import { DesignSystemContext } from '@databricks/design-system';
import { useContext } from 'react';

/**
 * Wrapped <Cascader /> component that ensures that it uses
 * design system context's popup container if its available.
 */
export const DuboisCascader = (props: CascaderProps) => {
  const designSystemContext = useContext(DesignSystemContext);
  if (designSystemContext) {
    return <Cascader getPopupContainer={designSystemContext.getPopupContainer} {...props} />;
  }

  return <Cascader {...props} />;
};
