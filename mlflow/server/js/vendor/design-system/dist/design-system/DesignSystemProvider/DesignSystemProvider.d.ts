import type { CSSObject } from '@emotion/react';
import type { ConfigProviderProps as AntDConfigProviderProps } from 'antd/lib/config-provider';
import React from 'react';
import type { DesignSystemFlags } from '../../flags';
export interface DesignSystemThemeProviderProps {
    isDarkMode?: boolean;
}
export interface DesignSystemProviderProps {
    /**
     * Set this to `true` to enable CSS animations for all Du Bois components inside this provider.
     * This is purely a visual enhancement, and is not necessary for the functionality of any component.
     */
    enableAnimation?: boolean;
    /** Use this to configure the base zIndex for Du Bois. We use this to stack our internal elements
     * correctly, and to ensure that our internal elements are always on top of any other elements. This
     * should be a number >= 1000.
     */
    zIndexBase?: number;
    /**
     * Use this to allow the application to provide the element used as a container for all "popup" components.
     */
    getPopupContainer?: () => HTMLElement;
    /** Set any of the `AvailableDesignSystemFlags` to `true` to enable that flag. */
    flags?: DesignSystemFlags;
    /** Whether to disable virtualization of legacy AntD components. Defaults to true in tests. */
    disableLegacyAntVirtualization?: boolean;
}
export interface DuboisThemeContextType {
    isDarkMode: boolean;
}
export interface DuboisContextType {
    enableAnimation: boolean;
    getPrefixCls: (suffix?: string) => string;
    getPopupContainer?: () => HTMLElement;
    flags: DesignSystemFlags;
}
export declare const DesignSystemThemeContext: React.Context<DuboisThemeContextType>;
export declare const DesignSystemContext: React.Context<DuboisContextType>;
export declare const DU_BOIS_ENABLE_ANIMATION_CLASSNAME = "du-bois-enable-animation";
export declare function getAnimationCss(enableAnimation: boolean): CSSObject;
type AntDConfigProviderPropsProps = Pick<AntDConfigProviderProps, 'prefixCls' | 'getPopupContainer' | 'virtual'>;
/** Only to be accessed by SupportsDuBoisThemes, except for special exceptions like tests and storybook. Ask in #dubois first if you need to use it. */
export declare const DesignSystemThemeProvider: React.FC<DesignSystemThemeProviderProps>;
export declare const DesignSystemProvider: React.FC<DesignSystemProviderProps>;
export declare const ApplyDesignSystemContextOverrides: React.FC<DesignSystemProviderProps>;
export declare const ApplyDesignSystemFlags: React.FC<{
    flags: DesignSystemFlags;
}>;
export interface DesignSystemAntDConfigProviderProps {
    children: React.ReactNode;
}
export declare const DesignSystemAntDConfigProvider: ({ children }: DesignSystemAntDConfigProviderProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export declare const useAntDConfigProviderContext: () => AntDConfigProviderPropsProps;
/**
 * When using AntD components inside Design System wrapper components (e.g. Modal, Collapse etc),
 * we don't want Design System's prefix class to override them.
 *
 * Since all Design System's components have are wrapped with DesignSystemAntDConfigProvider,
 * this won't affect their own prefixCls, but just allow nested antd components to keep their ant prefix.
 */
export declare const RestoreAntDDefaultClsPrefix: ({ children }: {
    children: React.ReactNode;
}) => import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=DesignSystemProvider.d.ts.map