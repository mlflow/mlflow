import type { LayoutProps as AntDLayoutProps } from 'antd';
import { Layout as AntDLayout } from 'antd';
import React from 'react';

import { DesignSystemAntDConfigProvider, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import type { DangerouslySetAntdProps } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

const { Header, Footer, Sider, Content } = AntDLayout;

export interface LayoutProps extends AntDLayoutProps, DangerouslySetAntdProps<AntDLayoutProps> {}

interface LayoutInterface extends React.FC<LayoutProps> {
  Header: typeof AntDLayout.Header;
  Footer: typeof AntDLayout.Footer;
  Sider: typeof AntDLayout.Sider;
  Content: typeof AntDLayout.Content;
}

/**
 * @deprecated Use PageWrapper instead
 */
export const Layout: LayoutInterface = /* #__PURE__ */ (() => {
  const Layout: LayoutInterface = ({ children, dangerouslySetAntdProps, ...props }) => {
    return (
      <DesignSystemAntDConfigProvider>
        <AntDLayout {...addDebugOutlineIfEnabled()} {...props} {...dangerouslySetAntdProps}>
          <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
        </AntDLayout>
      </DesignSystemAntDConfigProvider>
    );
  };

  Layout.Header = ({ children, ...props }) => (
    <DesignSystemAntDConfigProvider>
      <Header {...addDebugOutlineIfEnabled()} {...props}>
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </Header>
    </DesignSystemAntDConfigProvider>
  );
  Layout.Footer = ({ children, ...props }) => (
    <DesignSystemAntDConfigProvider>
      <Footer {...addDebugOutlineIfEnabled()} {...props}>
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </Footer>
    </DesignSystemAntDConfigProvider>
  );
  Layout.Sider = React.forwardRef(({ children, ...props }, ref) => (
    <DesignSystemAntDConfigProvider>
      <Sider {...addDebugOutlineIfEnabled()} {...props} ref={ref}>
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </Sider>
    </DesignSystemAntDConfigProvider>
  ));
  Layout.Content = ({ children, ...props }) => (
    <DesignSystemAntDConfigProvider>
      <Content {...addDebugOutlineIfEnabled()} {...props}>
        <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
      </Content>
    </DesignSystemAntDConfigProvider>
  );

  return Layout;
})();
