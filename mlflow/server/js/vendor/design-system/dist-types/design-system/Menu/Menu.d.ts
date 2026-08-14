import type { MenuProps as AntDMenuProps, MenuItemProps as AntDMenuItemProps } from 'antd';
import { Menu as AntDMenu } from 'antd';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export interface MenuProps extends AntDMenuProps, DangerouslySetAntdProps<AntDMenuProps>, HTMLDataAttributes {
}
export interface MenuItemProps extends AntDMenuItemProps {
}
interface MenuInterface extends React.FC<MenuProps> {
    Item: typeof AntDMenu.Item;
    ItemGroup: typeof AntDMenu.ItemGroup;
    SubMenu: typeof AntDMenu.SubMenu;
}
/**
 * @deprecated Use `DropdownMenu` instead.
 */
export declare const Menu: MenuInterface;
export {};
//# sourceMappingURL=Menu.d.ts.map