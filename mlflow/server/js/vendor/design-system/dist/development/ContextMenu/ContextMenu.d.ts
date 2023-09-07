/// <reference types="react" />
import type { Theme } from '@emotion/react';
import type { ContextMenuContentProps as RadixContextMenuContentProps, ContextMenuSubContentProps as RadixContextMenuSubContentProps, ContextMenuItemProps as RadixContextMenuItemProps, ContextMenuCheckboxItemProps as RadixContextMenuCheckboxItemProps, ContextMenuRadioItemProps as RadixContextMenuRadioItemProps, ContextMenuSubTriggerProps as RadixContextMenuSubTriggerProps, ContextMenuLabelProps as RadixContextMenuLabelProps } from '@radix-ui/react-context-menu';
export declare const Root: import("react").FC<import("@radix-ui/react-context-menu").ContextMenuProps>;
export declare const Trigger: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuTriggerProps & import("react").RefAttributes<HTMLSpanElement>>;
export declare const ItemIndicator: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuItemIndicatorProps & import("react").RefAttributes<HTMLSpanElement>>;
export declare const Group: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuGroupProps & import("react").RefAttributes<HTMLDivElement>>;
export declare const RadioGroup: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuRadioGroupProps & import("react").RefAttributes<HTMLDivElement>>;
export declare const Arrow: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuArrowProps & import("react").RefAttributes<SVGSVGElement>>;
export declare const Sub: import("react").FC<import("@radix-ui/react-context-menu").ContextMenuSubProps>;
export interface ContextMenuSubTriggerProps extends RadixContextMenuSubTriggerProps {
}
export declare const SubTrigger: ({ children, ...props }: ContextMenuSubTriggerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuContentProps extends RadixContextMenuContentProps {
    minWidth?: number;
}
export declare const Content: ({ children, minWidth, ...childrenProps }: ContextMenuContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuSubContentProps extends RadixContextMenuSubContentProps {
    minWidth?: number;
}
export declare const SubContent: ({ children, minWidth, ...childrenProps }: ContextMenuSubContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuItemProps extends RadixContextMenuItemProps {
}
export declare const Item: ({ children, ...props }: ContextMenuItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuCheckboxItemProps extends RadixContextMenuCheckboxItemProps {
}
export declare const CheckboxItem: ({ children, ...props }: ContextMenuCheckboxItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuRadioItemProps extends RadixContextMenuRadioItemProps {
}
export declare const RadioItem: ({ children, ...props }: ContextMenuRadioItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContextMenuLabelProps extends RadixContextMenuLabelProps {
}
export declare const Label: ({ children, ...props }: ContextMenuLabelProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const Hint: ({ children }: {
    children: React.ReactNode;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const Separator: () => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const itemIndicatorStyles: (theme: Theme) => import("@emotion/utils").SerializedStyles;
export declare const ContextMenu: {
    Root: import("react").FC<import("@radix-ui/react-context-menu").ContextMenuProps>;
    Trigger: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuTriggerProps & import("react").RefAttributes<HTMLSpanElement>>;
    Label: ({ children, ...props }: ContextMenuLabelProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Item: ({ children, ...props }: ContextMenuItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Group: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuGroupProps & import("react").RefAttributes<HTMLDivElement>>;
    RadioGroup: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuRadioGroupProps & import("react").RefAttributes<HTMLDivElement>>;
    CheckboxItem: ({ children, ...props }: ContextMenuCheckboxItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    RadioItem: ({ children, ...props }: ContextMenuRadioItemProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Arrow: import("react").ForwardRefExoticComponent<import("@radix-ui/react-context-menu").ContextMenuArrowProps & import("react").RefAttributes<SVGSVGElement>>;
    Separator: () => import("@emotion/react/jsx-runtime").JSX.Element;
    Sub: import("react").FC<import("@radix-ui/react-context-menu").ContextMenuSubProps>;
    SubTrigger: ({ children, ...props }: ContextMenuSubTriggerProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    SubContent: ({ children, minWidth, ...childrenProps }: ContextMenuSubContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Content: ({ children, minWidth, ...childrenProps }: ContextMenuContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
    Hint: ({ children }: {
        children: React.ReactNode;
    }) => import("@emotion/react/jsx-runtime").JSX.Element;
};
//# sourceMappingURL=ContextMenu.d.ts.map