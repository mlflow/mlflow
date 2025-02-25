import { ClassNames } from '@emotion/react';
import type {
  MenuProps as AntDMenuProps,
  MenuItemProps as AntDMenuItemProps,
  SubMenuProps as AntDSubmenuProps,
} from 'antd';
import { Menu as AntDMenu } from 'antd';

import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface MenuProps extends AntDMenuProps, DangerouslySetAntdProps<AntDMenuProps>, HTMLDataAttributes {}
export interface MenuItemProps extends AntDMenuItemProps {}

interface MenuInterface extends React.FC<MenuProps> {
  Item: typeof AntDMenu.Item;
  ItemGroup: typeof AntDMenu.ItemGroup;
  SubMenu: typeof AntDMenu.SubMenu;
}

/**
 * @deprecated Use `DropdownMenu` instead.
 */
export const Menu: MenuInterface = /* #__PURE__ */ (() => {
  const Menu: MenuInterface = ({ dangerouslySetAntdProps, ...props }) => {
    return (
      <DesignSystemAntDConfigProvider>
        <AntDMenu {...addDebugOutlineIfEnabled()} {...props} {...dangerouslySetAntdProps} />
      </DesignSystemAntDConfigProvider>
    );
  };

  Menu.Item = AntDMenu.Item;
  Menu.ItemGroup = AntDMenu.ItemGroup;
  Menu.SubMenu = function SubMenu({ dangerouslySetAntdProps, ...props }: SubMenuProps) {
    const { theme } = useDesignSystemTheme();

    return (
      <ClassNames>
        {({ css }) => {
          return (
            <AntDMenu.SubMenu
              {...addDebugOutlineIfEnabled()}
              popupClassName={css({
                zIndex: theme.options.zIndexBase + 50,
              })}
              popupOffset={[-6, -10]}
              {...props}
              {...dangerouslySetAntdProps}
            />
          );
        }}
      </ClassNames>
    );
  };

  return Menu;
})();

interface SubMenuProps extends AntDSubmenuProps, DangerouslySetAntdProps<AntDSubmenuProps> {}
