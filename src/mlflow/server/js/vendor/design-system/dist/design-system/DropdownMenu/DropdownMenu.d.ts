import type { CSSObject, Interpolation } from '@emotion/react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import React from 'react';
import type { Theme } from '../../theme';
export declare const Root: React.FC<DropdownMenu.DropdownMenuProps>;
export interface DropdownMenuProps extends DropdownMenu.MenuContentProps {
    minWidth?: number;
    isInsideModal?: boolean;
}
export interface DropdownMenuItemProps extends DropdownMenu.DropdownMenuItemProps {
    danger?: boolean;
    disabledReason?: React.ReactNode;
}
export interface DropdownMenuSubTriggerProps extends DropdownMenu.DropdownMenuSubTriggerProps {
    disabledReason?: React.ReactNode;
}
export interface DropdownMenuCheckboxItemProps extends DropdownMenu.DropdownMenuCheckboxItemProps {
    disabledReason?: React.ReactNode;
}
export interface DropdownMenuRadioItemProps extends DropdownMenu.DropdownMenuRadioItemProps {
    disabledReason?: React.ReactNode;
}
export declare const Content: React.ForwardRefExoticComponent<DropdownMenuProps & React.RefAttributes<HTMLDivElement>>;
export declare const SubContent: React.ForwardRefExoticComponent<Omit<DropdownMenuProps, "isInsideModal"> & React.RefAttributes<HTMLDivElement>>;
export declare const Trigger: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuTriggerProps & React.RefAttributes<HTMLButtonElement>>;
export declare const Item: React.ForwardRefExoticComponent<DropdownMenuItemProps & React.RefAttributes<HTMLDivElement>>;
export declare const Label: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuLabelProps & React.RefAttributes<HTMLDivElement>>;
export declare const Separator: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuSeparatorProps & React.RefAttributes<HTMLDivElement>>;
export declare const SubTrigger: React.ForwardRefExoticComponent<DropdownMenuSubTriggerProps & React.RefAttributes<HTMLDivElement>>;
/**
 * Deprecated. Use `SubTrigger` instead.
 * @deprecated
 */
export declare const TriggerItem: React.ForwardRefExoticComponent<DropdownMenuSubTriggerProps & React.RefAttributes<HTMLDivElement>>;
export declare const CheckboxItem: React.ForwardRefExoticComponent<DropdownMenuCheckboxItemProps & React.RefAttributes<HTMLDivElement>>;
export declare const ItemIndicator: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuItemIndicatorProps & React.RefAttributes<HTMLDivElement>>;
export declare const Arrow: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuArrowProps & React.RefAttributes<SVGSVGElement>>;
export declare const RadioItem: React.ForwardRefExoticComponent<DropdownMenuRadioItemProps & React.RefAttributes<HTMLDivElement>>;
export declare const Group: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuGroupProps & React.RefAttributes<HTMLDivElement>>;
export declare const RadioGroup: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuRadioGroupProps & React.RefAttributes<HTMLDivElement>>;
export declare const Sub: React.FC<DropdownMenu.DropdownMenuSubProps>;
export declare const HintColumn: React.ForwardRefExoticComponent<Pick<Pick<React.DetailedHTMLProps<React.HTMLAttributes<HTMLDivElement>, HTMLDivElement>, "key" | keyof React.HTMLAttributes<HTMLDivElement>> & {
    ref?: ((instance: HTMLDivElement | null) => void) | React.RefObject<HTMLDivElement> | null | undefined;
}, "key" | keyof React.HTMLAttributes<HTMLDivElement>> & React.RefAttributes<HTMLDivElement>>;
export declare const HintRow: React.ForwardRefExoticComponent<Pick<Pick<React.DetailedHTMLProps<React.HTMLAttributes<HTMLDivElement>, HTMLDivElement>, "key" | keyof React.HTMLAttributes<HTMLDivElement>> & {
    ref?: ((instance: HTMLDivElement | null) => void) | React.RefObject<HTMLDivElement> | null | undefined;
}, "key" | keyof React.HTMLAttributes<HTMLDivElement>> & React.RefAttributes<HTMLDivElement>>;
export declare const IconWrapper: React.ForwardRefExoticComponent<Pick<Pick<React.DetailedHTMLProps<React.HTMLAttributes<HTMLDivElement>, HTMLDivElement>, "key" | keyof React.HTMLAttributes<HTMLDivElement>> & {
    ref?: ((instance: HTMLDivElement | null) => void) | React.RefObject<HTMLDivElement> | null | undefined;
}, "key" | keyof React.HTMLAttributes<HTMLDivElement>> & React.RefAttributes<HTMLDivElement>>;
export declare const dropdownContentStyles: (theme: Theme) => CSSObject;
export declare const dropdownItemStyles: (theme: Theme) => Interpolation<Theme>;
export declare const dropdownSeparatorStyles: (theme: Theme) => {
    height: number;
    margin: string;
    backgroundColor: string;
};
//# sourceMappingURL=DropdownMenu.d.ts.map