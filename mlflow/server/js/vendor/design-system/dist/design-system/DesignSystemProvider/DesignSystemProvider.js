import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { ThemeProvider as EmotionThemeProvider } from '@emotion/react';
import { TooltipProvider as RadixTooltipProvider } from '@radix-ui/react-tooltip';
import { ConfigProvider as AntDConfigProvider, notification } from 'antd';
import React, { createContext, useContext, useEffect, useMemo } from 'react';
import { getTheme } from '../../theme';
import { getClassNamePrefix, getPrefixedClassNameFromTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags } from '../utils';
const DuboisContextDefaults = {
    enableAnimation: false,
    // Prefer to use useDesignSystemTheme.getPrefixedClassName instead
    getPrefixCls: (suffix) => (suffix ? `du-bois-${suffix}` : 'du-bois'),
    flags: {},
};
export const DesignSystemThemeContext = createContext({ isDarkMode: false });
export const DesignSystemContext = createContext(DuboisContextDefaults);
export const DU_BOIS_ENABLE_ANIMATION_CLASSNAME = 'du-bois-enable-animation';
export function getAnimationCss(enableAnimation) {
    const disableAnimationCss = {
        animationDuration: '0s !important',
        transition: 'none !important',
    };
    return enableAnimation
        ? {}
        : {
            // Apply to the current element
            ...disableAnimationCss,
            '&::before': disableAnimationCss,
            '&::after': disableAnimationCss,
            // Also apply to all child elements with a class that starts with our prefix
            [`[class*=du-bois]:not(.${DU_BOIS_ENABLE_ANIMATION_CLASSNAME}, .${DU_BOIS_ENABLE_ANIMATION_CLASSNAME} *)`]: {
                ...disableAnimationCss,
                // Also target any pseudo-elements associated with those elements, since these can also be animated.
                '&::before': disableAnimationCss,
                '&::after': disableAnimationCss,
            },
        };
}
const DesignSystemProviderPropsContext = React.createContext(null);
const AntDConfigProviderPropsContext = React.createContext(null);
/** Only to be accessed by SupportsDuBoisThemes, except for special exceptions like tests and storybook. Ask in #dubois first if you need to use it. */
export const DesignSystemThemeProvider = ({ isDarkMode = false, children, }) => {
    return _jsx(DesignSystemThemeContext.Provider, { value: { isDarkMode }, children: children });
};
export const DesignSystemProvider = ({ children, enableAnimation = false, zIndexBase = 1000, getPopupContainer, flags = {}, 
// Disable virtualization of legacy AntD components when running tests so that all items are rendered
disableLegacyAntVirtualization = process.env.NODE_ENV === 'test' ? true : undefined, }) => {
    const { isDarkMode } = useContext(DesignSystemThemeContext);
    const { useNewBorderColors } = useDesignSystemSafexFlags();
    const theme = useMemo(() => getTheme(isDarkMode, {
        enableAnimation,
        zIndexBase,
        useNewBorderColors,
    }), 
    // TODO: revisit this
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isDarkMode, zIndexBase, useNewBorderColors]);
    const providerPropsContext = useMemo(() => ({
        isDarkMode,
        enableAnimation,
        zIndexBase,
        getPopupContainer,
        flags,
    }), [isDarkMode, enableAnimation, zIndexBase, getPopupContainer, flags]);
    const classNamePrefix = getClassNamePrefix(theme);
    const value = useMemo(() => {
        return {
            enableAnimation,
            isDarkMode,
            getPrefixCls: (suffix) => getPrefixedClassNameFromTheme(theme, suffix),
            getPopupContainer,
            flags,
        };
    }, [enableAnimation, theme, isDarkMode, getPopupContainer, flags]);
    useEffect(() => {
        return () => {
            // when the design system context is unmounted, make sure the notification cache is also cleaned up
            notification.destroy();
        };
    }, []);
    return (_jsx(DesignSystemProviderPropsContext.Provider, { value: providerPropsContext, children: _jsx(EmotionThemeProvider, { theme: theme, children: _jsx(AntDConfigProviderPropsContext.Provider, { value: { prefixCls: classNamePrefix, getPopupContainer, virtual: !disableLegacyAntVirtualization }, children: _jsx(RadixTooltipProvider, { children: _jsx(DesignSystemContext.Provider, { value: value, children: children }) }) }) }) }));
};
export const ApplyDesignSystemContextOverrides = ({ enableAnimation, zIndexBase, getPopupContainer, flags, children, }) => {
    const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
    if (parentDesignSystemProviderProps === null) {
        throw new Error(`ApplyDesignSystemContextOverrides cannot be used standalone - DesignSystemProvider must exist in the React context`);
    }
    const newProps = useMemo(() => ({
        ...parentDesignSystemProviderProps,
        enableAnimation: enableAnimation ?? parentDesignSystemProviderProps.enableAnimation,
        zIndexBase: zIndexBase ?? parentDesignSystemProviderProps.zIndexBase,
        getPopupContainer: getPopupContainer ?? parentDesignSystemProviderProps.getPopupContainer,
        flags: {
            ...parentDesignSystemProviderProps.flags,
            ...flags,
        },
    }), [parentDesignSystemProviderProps, enableAnimation, zIndexBase, getPopupContainer, flags]);
    return _jsx(DesignSystemProvider, { ...newProps, children: children });
};
// This is a more-specific version of `ApplyDesignSystemContextOverrides` that only allows overriding the flags.
export const ApplyDesignSystemFlags = ({ flags, children }) => {
    const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
    if (parentDesignSystemProviderProps === null) {
        throw new Error(`ApplyDesignSystemFlags cannot be used standalone - DesignSystemProvider must exist in the React context`);
    }
    const newProps = useMemo(() => ({
        ...parentDesignSystemProviderProps,
        flags: {
            ...parentDesignSystemProviderProps.flags,
            ...flags,
        },
    }), [parentDesignSystemProviderProps, flags]);
    return _jsx(DesignSystemProvider, { ...newProps, children: children });
};
export const DesignSystemAntDConfigProvider = ({ children }) => {
    const antdContext = useAntDConfigProviderContext();
    return _jsx(AntDConfigProvider, { ...antdContext, children: children });
};
export const useAntDConfigProviderContext = () => {
    return useContext(AntDConfigProviderPropsContext) ?? { prefixCls: undefined };
};
/**
 * When using AntD components inside Design System wrapper components (e.g. Modal, Collapse etc),
 * we don't want Design System's prefix class to override them.
 *
 * Since all Design System's components have are wrapped with DesignSystemAntDConfigProvider,
 * this won't affect their own prefixCls, but just allow nested antd components to keep their ant prefix.
 */
export const RestoreAntDDefaultClsPrefix = ({ children }) => {
    return _jsx(AntDConfigProvider, { prefixCls: "ant", children: children });
};
//# sourceMappingURL=DesignSystemProvider.js.map