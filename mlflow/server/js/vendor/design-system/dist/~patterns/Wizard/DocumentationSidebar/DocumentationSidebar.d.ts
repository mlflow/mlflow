import type { PropsWithChildren } from 'react';
import React from 'react';
import type { TooltipProps } from '../../../design-system';
export type RootProps = PropsWithChildren<{
    /**
     * Initial content id to be displayed
     *
     * @default `undefined`
     */
    initialContentId?: string | undefined;
}>;
export declare function Root({ children, initialContentId }: RootProps): import("@emotion/react/jsx-runtime").JSX.Element;
export interface DocumentationSideBarState {
    currentContentId: string | undefined;
    setCurrentContentId: (value: string | undefined) => void;
}
export declare const useDocumentationSidebarContext: () => DocumentationSideBarState;
export interface TriggerProps<T extends string> extends Omit<TooltipProps, 'content' | 'children'> {
    /**
     * ContentId that will be passed along to the Content.
     */
    contentId: T;
    /**
     * aria-label for the info icon button
     */
    label?: string;
    /**
     * Optional content for tooltip that will wrap the trigger content
     */
    tooltipContent?: React.ReactNode;
    /**
     * Set to true if you want to render your own trigger; requires children to a valid single React node
     *
     * Will default to render an `InfoIcon` wrapped in a button
     */
    asChild?: boolean;
    /**
     * Will be rendered as trigger content if `asChild` is true
     *
     * The node must be an interactive element ex: <Button/>
     */
    children?: React.ReactNode;
}
export declare function Trigger<T extends string>({ contentId, label, tooltipContent, asChild, children, ...tooltipProps }: TriggerProps<T>): import("@emotion/react/jsx-runtime").JSX.Element;
export interface ContentChildProps<T extends string> {
    contentId: T;
}
export type ContentProps = {
    /**
     * @default 100%
     */
    width?: number;
    /**
     * This must be a single React element that takes in an optional `contentId: string` prop
     */
    children: React.ReactNode;
    /**
     * Title displayed atop for all content id
     */
    title: string;
    /**
     * The compact modal title; defaults to `title`
     */
    modalTitleWhenCompact?: string;
    /**
     * aria-label for the close button and a button label for the compact modal version
     */
    closeLabel: string;
    /**
     * If true the documentation content will display in a modal instead of a sidebar
     *
     * Example set to true for a specific breakpoint:
     * const displayModalWhenCompact = useMediaQuery({query: `(max-width: ${theme.responsive.breakpoints.lg }px)`})
     */
    displayModalWhenCompact: boolean;
};
export declare function Content<T extends string>({ title, modalTitleWhenCompact, width, children, closeLabel, displayModalWhenCompact, }: ContentProps): import("@emotion/react/jsx-runtime").JSX.Element | null;
//# sourceMappingURL=DocumentationSidebar.d.ts.map