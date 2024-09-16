import { type CSSObject, type Interpolation } from '@emotion/react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import type { ReactElement } from 'react';
import React from 'react';
import type { Theme } from '../../theme';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventProps, AnalyticsEventValueChangeNoPiiFlagProps } from '../types';
export declare const Root: ({ children, ...props }: DropdownMenu.DropdownMenuProps) => ReactElement;
export interface DropdownMenuProps extends DropdownMenu.MenuContentProps {
    minWidth?: number;
    forceCloseOnEscape?: boolean;
}
export interface DropdownMenuItemProps extends DropdownMenu.DropdownMenuItemProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
    danger?: boolean;
    disabledReason?: React.ReactNode;
}
export interface DropdownMenuSubTriggerProps extends DropdownMenu.DropdownMenuSubTriggerProps {
    disabledReason?: React.ReactNode;
}
export interface DropdownMenuCheckboxItemProps extends DropdownMenu.DropdownMenuCheckboxItemProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    disabledReason?: React.ReactNode;
}
export interface DropdownMenuRadioGroupProps extends DropdownMenu.DropdownMenuRadioGroupProps, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
}
export interface DropdownMenuRadioItemProps extends DropdownMenu.DropdownMenuRadioItemProps {
    disabledReason?: React.ReactNode;
}
export declare const Content: React.ForwardRefExoticComponent<DropdownMenuProps & React.RefAttributes<HTMLDivElement>>;
export declare const SubContent: React.ForwardRefExoticComponent<Omit<DropdownMenuProps, "forceCloseOnEscape"> & React.RefAttributes<HTMLDivElement>>;
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
export declare const RadioGroup: React.ForwardRefExoticComponent<DropdownMenuRadioGroupProps & React.RefAttributes<HTMLDivElement>>;
export declare const ItemIndicator: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuItemIndicatorProps & React.RefAttributes<HTMLDivElement>>;
export declare const Arrow: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuArrowProps & React.RefAttributes<SVGSVGElement>>;
export declare const RadioItem: React.ForwardRefExoticComponent<DropdownMenuRadioItemProps & React.RefAttributes<HTMLDivElement>>;
export declare const Sub: ({ children, onOpenChange, ...props }: DropdownMenu.DropdownMenuSubProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const Group: React.ForwardRefExoticComponent<DropdownMenu.DropdownMenuGroupProps & React.RefAttributes<HTMLDivElement>>;
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