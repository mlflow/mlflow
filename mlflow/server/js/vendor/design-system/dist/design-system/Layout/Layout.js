import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Layout as AntDLayout } from 'antd';
import React from 'react';
import { DesignSystemAntDConfigProvider, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const { Header, Footer, Sider, Content } = AntDLayout;
/**
 * @deprecated Use PageWrapper instead
 */
export const Layout = /* #__PURE__ */ (() => {
    const Layout = ({ children, dangerouslySetAntdProps, ...props }) => {
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDLayout, { ...addDebugOutlineIfEnabled(), ...props, ...dangerouslySetAntdProps, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
    };
    Layout.Header = ({ children, ...props }) => (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(Header, { ...addDebugOutlineIfEnabled(), ...props, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
    Layout.Footer = ({ children, ...props }) => (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(Footer, { ...addDebugOutlineIfEnabled(), ...props, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
    Layout.Sider = React.forwardRef(({ children, ...props }, ref) => (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(Sider, { ...addDebugOutlineIfEnabled(), ...props, ref: ref, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) })));
    Layout.Content = ({ children, ...props }) => (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(Content, { ...addDebugOutlineIfEnabled(), ...props, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
    return Layout;
})();
//# sourceMappingURL=Layout.js.map