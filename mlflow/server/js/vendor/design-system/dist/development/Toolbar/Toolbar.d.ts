import * as RadixToolbar from '@radix-ui/react-toolbar';
export type ToolbarRootProps = Omit<RadixToolbar.ToolbarProps, 'orientation'>;
export type ToolbarButtonProps = RadixToolbar.ToolbarButtonProps;
export type ToolbarSeparatorProps = RadixToolbar.ToolbarSeparatorProps;
export type ToolbarLinkProps = RadixToolbar.ToolbarLinkProps;
export type ToolbarToogleGroupProps = RadixToolbar.ToolbarToggleGroupSingleProps | RadixToolbar.ToolbarToggleGroupMultipleProps;
export type ToolbarToggleItemProps = RadixToolbar.ToolbarToggleItemProps;
export declare const Root: import("react").ForwardRefExoticComponent<ToolbarRootProps & import("react").RefAttributes<HTMLDivElement>>;
export declare const Button: import("react").ForwardRefExoticComponent<RadixToolbar.ToolbarButtonProps & import("react").RefAttributes<HTMLButtonElement>>;
export declare const Separator: import("react").ForwardRefExoticComponent<RadixToolbar.ToolbarSeparatorProps & import("react").RefAttributes<HTMLDivElement>>;
export declare const Link: import("react").ForwardRefExoticComponent<RadixToolbar.ToolbarLinkProps & import("react").RefAttributes<HTMLAnchorElement>>;
export declare const ToggleGroup: import("react").ForwardRefExoticComponent<(RadixToolbar.ToolbarToggleGroupSingleProps | RadixToolbar.ToolbarToggleGroupMultipleProps) & import("react").RefAttributes<HTMLDivElement>>;
export declare const ToggleItem: import("react").ForwardRefExoticComponent<RadixToolbar.ToolbarToggleItemProps & import("react").RefAttributes<HTMLButtonElement>>;
//# sourceMappingURL=Toolbar.d.ts.map