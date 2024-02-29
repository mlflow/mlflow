import type { Theme } from '@emotion/react';
import type { ContextMenuCheckboxItemProps as RadixContextMenuCheckboxItemProps, ContextMenuContentProps as RadixContextMenuContentProps, ContextMenuItemProps as RadixContextMenuItemProps, ContextMenuLabelProps as RadixContextMenuLabelProps, ContextMenuRadioItemProps as RadixContextMenuRadioItemProps, ContextMenuSubContentProps as RadixContextMenuSubContentProps, ContextMenuSubTriggerProps as RadixContextMenuSubTriggerProps } from '@radix-ui/react-context-menu';
import React from 'react';
export declare const Root: React.FC<import("@radix-ui/react-context-menu").ContextMenuProps>;
export declare const Trigger: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuTriggerProps & React.RefAttributes<HTMLSpanElement>>;
export declare const ItemIndicator: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuItemIndicatorProps & React.RefAttributes<HTMLSpanElement>>;
export declare const Group: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuGroupProps & React.RefAttributes<HTMLDivElement>>;
export declare const RadioGroup: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuRadioGroupProps & React.RefAttributes<HTMLDivElement>>;
export declare const Arrow: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuArrowProps & React.RefAttributes<SVGSVGElement>>;
export declare const Sub: React.FC<import("@radix-ui/react-context-menu").ContextMenuSubProps>;
export interface ContextMenuSubTriggerProps extends RadixContextMenuSubTriggerProps {
    disabledReason?: React.ReactNode;
}
export declare const SubTrigger: ({ children, disabledReason, ...props }: ContextMenuSubTriggerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuContentProps extends RadixContextMenuContentProps {
    minWidth?: number;
}
export declare const Content: ({ children, minWidth, ...childrenProps }: ContextMenuContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuSubContentProps extends RadixContextMenuSubContentProps {
    minWidth?: number;
}
export declare const SubContent: ({ children, minWidth, ...childrenProps }: ContextMenuSubContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuItemProps extends RadixContextMenuItemProps {
    disabledReason?: React.ReactNode;
}
export declare const Item: ({ children, disabledReason, ...props }: ContextMenuItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuCheckboxItemProps extends RadixContextMenuCheckboxItemProps {
    disabledReason?: React.ReactNode;
}
export declare const CheckboxItem: ({ children, disabledReason, ...props }: ContextMenuCheckboxItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
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
    Root: React.FC<import("@radix-ui/react-context-menu").ContextMenuProps>;
    Trigger: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuTriggerProps & React.RefAttributes<HTMLSpanElement>>;
    Label: ({ children, ...props }: ContextMenuLabelProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Item: ({ children, disabledReason, ...props }: ContextMenuItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Group: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuGroupProps & React.RefAttributes<HTMLDivElement>>;
    RadioGroup: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuRadioGroupProps & React.RefAttributes<HTMLDivElement>>;
    CheckboxItem: ({ children, disabledReason, ...props }: ContextMenuCheckboxItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    RadioItem: ({ children, disabledReason, ...props }: ContextMenuRadioItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Arrow: React.ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuArrowProps & React.RefAttributes<SVGSVGElement>>;
    Separator: () => import("@emotion/react/jsx-runtime").JSX.Element;
    Sub: React.FC<import("@radix-ui/react-context-menu").ContextMenuSubProps>;
    SubTrigger: ({ children, disabledReason, ...props }: ContextMenuSubTriggerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    SubContent: ({ children, minWidth, ...childrenProps }: ContextMenuSubContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Content: ({ children, minWidth, ...childrenProps }: ContextMenuContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Hint: ({ children }: {
        children: React.ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
};
//# sourceMappingURL=ContextMenu.d.ts.map