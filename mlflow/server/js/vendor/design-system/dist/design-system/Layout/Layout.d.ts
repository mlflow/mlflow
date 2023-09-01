import type { LayoutProps as AntDLayoutProps } from 'antd';
import { Layout as AntDLayout } from 'antd';
import React from 'react';
import type { DangerouslySetAntdProps } from '../types';
export interface LayoutProps extends AntDLayoutProps, DangerouslySetAntdProps<AntDLayoutProps> {
}
interface LayoutInterface extends React.FC<LayoutProps> {
    Header: typeof AntDLayout.Header;
    Footer: typeof AntDLayout.Footer;
    Sider: typeof AntDLayout.Sider;
    Content: typeof AntDLayout.Content;
}
/**
 * @deprecated Use PageWrapper instead
 */
export declare const Layout: LayoutInterface;
export {};
//# sourceMappingURL=Layout.d.ts.map