import * as DialogPrimitive from '@radix-ui/react-dialog';
import React from 'react';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventOptionalProps } from '../types';
export interface DrawerContentProps extends AnalyticsEventOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    /** Contents displayed in the drawer */
    children: React.ReactNode;
    /**
     * Drawer header to be announced when the dialog is opened
     * If passing in a string styling will be provided, otherwise caller is responsible for proper title styling
     **/
    title: React.ReactNode;
    /**
     * The content width with a min width of 320 and a max width of 90vw
     * @default 320
     */
    width?: number | string;
    /**
     * The layout direction, on which side the drawer will appear
     * @default 'right'
     */
    position?: 'left' | 'right';
    /**
     * Provide a footer; using this property will ensure the correct scrolling behavior
     * @default 'undefined'
     */
    footer?: React.ReactNode;
    /**
     * Delegates all content scroll behavior to the caller if true
     *    Disable the default scroll drop shadow
     *    Hide the vertical content overflow
     *    Sets content right padding to 0 to leave room for caller to do so for proper scrollbar placement
     * @default false
     */
    useCustomScrollBehavior?: boolean;
    /**
     * If true the content of the Drawer will take up all available vertical space.
     * This is to keep the footer at the bottom of the drawer
     * @default false
     */
    expandContentToFullHeight?: boolean;
    /**
     * Disable auto focus on open
     * @default false
     */
    disableOpenAutoFocus?: boolean;
    /**
     * If true, the drawer and the backdrop will both be hidden. They will remain mounted, but not visible.
     * @default false
     */
    seeThrough?: boolean;
    /**
     * Event handler called when an interaction (pointer or focus event) happens outside the bounds of the component.
     * It can be prevented by calling event.preventDefault.
     */
    onInteractOutside?: DialogPrimitive.DialogContentProps['onInteractOutside'];
    /**
     * If true, the "x" icon in the header will be hidden
     * @default false
     */
    hideClose?: boolean;
}
export declare const Content: ({ children, footer, title, width, position: positionOverride, useCustomScrollBehavior, expandContentToFullHeight, disableOpenAutoFocus, onInteractOutside, seeThrough, hideClose, componentId, analyticsEvents, }: DrawerContentProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export declare function Root(props: Pick<DialogPrimitive.DialogProps, 'onOpenChange' | 'children' | 'open' | 'modal'>): import("@emotion/react/jsx-runtime").JSX.Element;
export declare function Trigger(props: Omit<DialogPrimitive.DialogTriggerProps, 'asChild'>): import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Drawer.d.ts.map