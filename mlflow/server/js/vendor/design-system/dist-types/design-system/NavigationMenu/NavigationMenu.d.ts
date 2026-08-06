import * as RadixNavigationMenu from '@radix-ui/react-navigation-menu';
import React from 'react';
export interface RootProps extends Omit<RadixNavigationMenu.NavigationMenuProps, 'asChild' | 'defaultValue' | 'value' | 'onValueChange' | 'delayDuration' | 'skipDelayDuration' | 'dir' | 'orientation'> {
}
export declare const Root: React.ForwardRefExoticComponent<RootProps & React.RefAttributes<HTMLDivElement>>;
export interface ListProps extends Omit<RadixNavigationMenu.NavigationMenuListProps, 'asChild'> {
}
export declare const List: React.ForwardRefExoticComponent<ListProps & React.RefAttributes<HTMLUListElement>>;
export interface ItemProps extends Omit<RadixNavigationMenu.NavigationMenuItemProps, 'asChild' | 'value'>, Pick<RadixNavigationMenu.NavigationMenuLinkProps, 'active'> {
}
export declare const Item: React.ForwardRefExoticComponent<ItemProps & React.RefAttributes<HTMLLIElement>>;
//# sourceMappingURL=NavigationMenu.d.ts.map