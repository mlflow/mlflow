import type { Interpolation, Theme as EmotionTheme } from '@emotion/react';
import type { EmotionJSX } from '@storybook/theming/dist/ts3.9/_modules/@emotion-react-types-jsx-namespace';
import React, { type CSSProperties } from 'react';
import { type ButtonProps } from '../Button';
export interface SidebarProps {
    /** The layout direction */
    position?: 'left' | 'right';
    /** Contents displayed in the sidebar */
    children?: React.ReactNode;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export interface NavProps {
    /** Contents displayed in the nav bar */
    children?: React.ReactNode;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export interface NavButtonProps extends ButtonProps {
    /** Check if the currrent button in nav bar is being selected */
    active?: boolean;
    /** Check if the currrent button in nav bar is being disabled */
    disabled?: boolean;
    /** The icon on the button */
    icon?: EmotionJSX.Element;
    /** Contents displayed in the nav bar */
    children?: React.ReactNode;
    /** The callback function when nav button is clicked */
    onClick?: () => void;
    'aria-label'?: string;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export interface ContentProps {
    /** The open panel id */
    openPanelId?: number;
    /** The content width, default is 200px */
    width?: number;
    /** The minimum content width */
    minWidth?: number;
    /** The maximum content width */
    maxWidth?: number;
    /** Whether or not to make the component resizable */
    disableResize?: boolean;
    /** Whether or not to show a close button which can close the panel */
    closable?: boolean;
    /** Whether to destroy inactive panels and their state when initializing the component or switching the active panel */
    destroyInactivePanels?: boolean;
    /** The callback function when close button is clicked */
    onClose?: () => void;
    /** This callback function is called when the content resize is started */
    onResizeStart?: (size: number) => void;
    /** This callback function is called when the content resize is completed */
    onResizeStop?: (size: number) => void;
    /** Contents displayed in the content */
    children?: React.ReactNode;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
    /** Applies styles to the react-resizable container */
    resizeBoxStyle?: CSSProperties;
}
export interface PanelProps {
    /** The panel id */
    panelId: number;
    /** Contents displayed in the the panel */
    children?: React.ReactNode;
    /** Forced render of content in the panel, not lazy render after clicking */
    forceRender?: boolean;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export interface PanelHeaderProps {
    /** Contents displayed in the header section of the panel */
    children?: React.ReactNode;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export interface PanelHeaderTitleProps {
    /** Text displayed in the header section of the panel */
    title: string;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export interface PanelHeaderButtonProps {
    /** Optional buttons displayed in the panel header */
    children?: React.ReactNode;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export interface PanelBodyProps {
    /** Contents displayed in the body of the panel */
    children?: React.ReactNode;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
export declare function Nav({ children, dangerouslyAppendEmotionCSS }: NavProps): JSX.Element;
export declare function NavButton({ active, disabled, icon, onClick, children, dangerouslyAppendEmotionCSS, 'aria-label': ariaLabel, ...restProps }: NavButtonProps): JSX.Element;
export declare function Content({ disableResize, openPanelId, closable, onClose, onResizeStart, onResizeStop, width, minWidth, maxWidth, destroyInactivePanels, children, dangerouslyAppendEmotionCSS, resizeBoxStyle, }: ContentProps): JSX.Element;
export declare function Panel({ panelId, children, forceRender, dangerouslyAppendEmotionCSS, ...delegated }: PanelProps): JSX.Element | null;
export declare function PanelHeader({ children, dangerouslyAppendEmotionCSS }: PanelHeaderProps): JSX.Element;
export declare function PanelHeaderTitle({ title, dangerouslyAppendEmotionCSS }: PanelHeaderTitleProps): import("@emotion/react/jsx-runtime").JSX.Element;
export declare function PanelHeaderButtons({ children, dangerouslyAppendEmotionCSS }: PanelHeaderButtonProps): import("@emotion/react/jsx-runtime").JSX.Element;
export declare function PanelBody({ children, dangerouslyAppendEmotionCSS }: PanelBodyProps): JSX.Element;
export declare const Sidebar: {
    ({ position, children, dangerouslyAppendEmotionCSS }: SidebarProps): JSX.Element;
    Content: typeof Content;
    Nav: typeof Nav;
    NavButton: typeof NavButton;
    Panel: typeof Panel;
    PanelHeader: typeof PanelHeader;
    PanelHeaderTitle: typeof PanelHeaderTitle;
    PanelHeaderButtons: typeof PanelHeaderButtons;
    PanelBody: typeof PanelBody;
};
//# sourceMappingURL=Sidebar.d.ts.map