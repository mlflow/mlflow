import type { Theme } from '@emotion/react';
import type { ContextMenuCheckboxItemProps as RadixContextMenuCheckboxItemProps, ContextMenuContentProps as RadixContextMenuContentProps, ContextMenuItemProps as RadixContextMenuItemProps, ContextMenuLabelProps as RadixContextMenuLabelProps, ContextMenuRadioGroupProps as RadixContextMenuRadioGroupProps, ContextMenuRadioItemProps as RadixContextMenuRadioItemProps, ContextMenuSubContentProps as RadixContextMenuSubContentProps, ContextMenuSubTriggerProps as RadixContextMenuSubTriggerProps, ContextMenuProps as RadixContextMenuProps } from '@radix-ui/react-context-menu';
import type { ReactElement } from 'react';
import React from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '..';
import type { AnalyticsEventProps, AnalyticsEventValueChangeNoPiiFlagProps } from '../types';
export declare const Trigger: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuTriggerProps & React.RefAttributes<HTMLSpanElement>>;
export declare const ItemIndicator: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuItemIndicatorProps & React.RefAttributes<HTMLSpanElement>>;
export declare const Group: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuGroupProps & React.RefAttributes<HTMLDivElement>>;
export declare const Arrow: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuArrowProps & React.RefAttributes<SVGSVGElement>>;
export declare const Sub: React.FC<import("@radix-ui/react-context-menu").ContextMenuSubProps>;
export declare const Root: ({ children, onOpenChange, ...props }: RadixContextMenuProps) => ReactElement;
export interface ContextMenuSubTriggerProps extends RadixContextMenuSubTriggerProps {
    disabledReason?: React.ReactNode;
    withChevron?: boolean;
}
export declare const SubTrigger: ({ children, disabledReason, withChevron, ...props }: ContextMenuSubTriggerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuContentProps extends RadixContextMenuContentProps {
    minWidth?: number;
    forceCloseOnEscape?: boolean;
}
export declare const Content: ({ children, minWidth, forceCloseOnEscape, onEscapeKeyDown, onKeyDown, ...childrenProps }: ContextMenuContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuSubContentProps extends RadixContextMenuSubContentProps {
    minWidth?: number;
}
export declare const SubContent: ({ children, minWidth, ...childrenProps }: ContextMenuSubContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuItemProps extends RadixContextMenuItemProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
    disabledReason?: React.ReactNode;
}
export declare const Item: ({ children, disabledReason, onClick, componentId, analyticsEvents, asChild, ...props }: ContextMenuItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuCheckboxItemProps extends RadixContextMenuCheckboxItemProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    disabledReason?: React.ReactNode;
}
export declare const CheckboxItem: ({ children, disabledReason, onCheckedChange, componentId, analyticsEvents, ...props }: ContextMenuCheckboxItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuRadioGroupProps extends RadixContextMenuRadioGroupProps, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
}
export declare const RadioGroup: ({ onValueChange, componentId, analyticsEvents, valueHasNoPii, ...props }: ContextMenuRadioGroupProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuRadioItemProps extends RadixContextMenuRadioItemProps {
    disabledReason?: React.ReactNode;
}
export declare const RadioItem: ({ children, disabledReason, ...props }: ContextMenuRadioItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuLabelProps extends RadixContextMenuLabelProps {
}
export declare const Label: ({ children, ...props }: ContextMenuLabelProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const Hint: ({ children }: {
    children: React.ReactNode;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const Separator: () => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const itemIndicatorStyles: (theme: Theme) => import("@emotion/utils").SerializedStyles;
export declare const ContextMenu: {
    Root: ({ children, onOpenChange, ...props }: RadixContextMenuProps) => ReactElement;
    Trigger: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuTriggerProps & React.RefAttributes<HTMLSpanElement>>;
    Label: ({ children, ...props }: ContextMenuLabelProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Item: ({ children, disabledReason, onClick, componentId, analyticsEvents, asChild, ...props }: ContextMenuItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Group: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuGroupProps & React.RefAttributes<HTMLDivElement>>;
    RadioGroup: ({ onValueChange, componentId, analyticsEvents, valueHasNoPii, ...props }: ContextMenuRadioGroupProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    CheckboxItem: ({ children, disabledReason, onCheckedChange, componentId, analyticsEvents, ...props }: ContextMenuCheckboxItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    RadioItem: ({ children, disabledReason, ...props }: ContextMenuRadioItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Arrow: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuArrowProps & React.RefAttributes<SVGSVGElement>>;
    Separator: () => import("@emotion/react/jsx-runtime").JSX.Element;
    Sub: React.FC<import("@radix-ui/react-context-menu").ContextMenuSubProps>;
    SubTrigger: ({ children, disabledReason, withChevron, ...props }: ContextMenuSubTriggerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    SubContent: ({ children, minWidth, ...childrenProps }: ContextMenuSubContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Content: ({ children, minWidth, forceCloseOnEscape, onEscapeKeyDown, onKeyDown, ...childrenProps }: ContextMenuContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Hint: ({ children }: {
        children: React.ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
};
//# sourceMappingURL=ContextMenu.d.ts.map