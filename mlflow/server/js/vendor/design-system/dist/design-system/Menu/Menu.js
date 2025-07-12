import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { ClassNames } from '@emotion/react';
import { Menu as AntDMenu } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';
/**
 * @deprecated Use `DropdownMenu` instead.
 */
export const Menu = /* #__PURE__ */ (() => {
    const Menu = ({ dangerouslySetAntdProps, ...props }) => {
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDMenu, { ...addDebugOutlineIfEnabled(), ...props, ...dangerouslySetAntdProps }) }));
    };
    Menu.Item = AntDMenu.Item;
    Menu.ItemGroup = AntDMenu.ItemGroup;
    Menu.SubMenu = function SubMenu({ dangerouslySetAntdProps, ...props }) {
        const { theme } = useDesignSystemTheme();
        return (_jsx(ClassNames, { children: ({ css }) => {
                return (_jsx(AntDMenu.SubMenu, { ...addDebugOutlineIfEnabled(), popupClassName: css({
                        zIndex: theme.options.zIndexBase + 50,
                    }), popupOffset: [-6, -10], ...props, ...dangerouslySetAntdProps }));
            } }));
    };
    return Menu;
})();
//# sourceMappingURL=Menu.js.map