import React from 'react';
import type { ContentProps, NavButtonProps, NavProps, PanelBodyProps, PanelHeaderButtonProps, PanelHeaderProps, PanelHeaderTitleProps, PanelProps, SidebarProps } from './types';
export declare function Nav({ children, dangerouslyAppendEmotionCSS }: NavProps): JSX.Element;
export declare const NavButton: React.ForwardRefExoticComponent<NavButtonProps & React.RefAttributes<HTMLButtonElement>>;
export declare const Content: React.ForwardRefExoticComponent<ContentProps & React.RefAttributes<HTMLDivElement>>;
export declare function Panel({ panelId, children, forceRender, dangerouslyAppendEmotionCSS, ...delegated }: PanelProps): JSX.Element | null;
export declare function PanelHeader({ children, dangerouslyAppendEmotionCSS, componentId, closeIcon }: PanelHeaderProps): JSX.Element;
export declare function PanelHeaderTitle({ title, dangerouslyAppendEmotionCSS }: PanelHeaderTitleProps): import("@emotion/react/jsx-runtime").JSX.Element;
export declare function PanelHeaderButtons({ children, dangerouslyAppendEmotionCSS }: PanelHeaderButtonProps): import("@emotion/react/jsx-runtime").JSX.Element;
export declare function PanelBody({ children, dangerouslyAppendEmotionCSS }: PanelBodyProps): JSX.Element;
export declare const Sidebar: {
    ({ position, children, dangerouslyAppendEmotionCSS, ...dataProps }: SidebarProps): JSX.Element;
    Content: React.ForwardRefExoticComponent<ContentProps & React.RefAttributes<HTMLDivElement>>;
    Nav: typeof Nav;
    NavButton: React.ForwardRefExoticComponent<NavButtonProps & React.RefAttributes<HTMLButtonElement>>;
    Panel: typeof Panel;
    PanelHeader: typeof PanelHeader;
    PanelHeaderTitle: typeof PanelHeaderTitle;
    PanelHeaderButtons: typeof PanelHeaderButtons;
    PanelBody: typeof PanelBody;
};
//# sourceMappingURL=Sidebar.d.ts.map