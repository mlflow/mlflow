import type { CSSObject } from '@emotion/react';
import { ThemeProvider as EmotionThemeProvider } from '@emotion/react';
import { TooltipProvider as RadixTooltipProvider } from '@radix-ui/react-tooltip';
import { ConfigProvider as AntDConfigProvider, notification } from 'antd';
import type { ConfigProviderProps as AntDConfigProviderProps } from 'antd/lib/config-provider';
import React, { createContext, useContext, useEffect, useMemo } from 'react';

import type { DesignSystemFlags } from '../../flags';
import { getTheme } from '../../theme';
import { getClassNamePrefix, getPrefixedClassNameFromTheme } from '../Hooks/useDesignSystemTheme';

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
  // Prefer to use useDesignSystemTheme.getPrefixedClassName instead
  // TODO(sschneider): Replace in all components and possibly in webapp.
  getPrefixCls: (suffix?: string) => string;
  getPopupContainer?: () => HTMLElement;
  flags: DesignSystemFlags;
}

const DuboisContextDefaults: DuboisContextType = {
  enableAnimation: false,
  // Prefer to use useDesignSystemTheme.getPrefixedClassName instead
  getPrefixCls: (suffix?: string) => (suffix ? `du-bois-${suffix}` : 'du-bois'),
  flags: {},
};

export const DesignSystemThemeContext = createContext<DuboisThemeContextType>({ isDarkMode: false });

export const DesignSystemContext = createContext<DuboisContextType>(DuboisContextDefaults);

export const DU_BOIS_ENABLE_ANIMATION_CLASSNAME = 'du-bois-enable-animation';

export function getAnimationCss(enableAnimation: boolean): CSSObject {
  const disableAnimationCss: CSSObject = {
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

const DesignSystemProviderPropsContext = React.createContext<DesignSystemProviderProps | null>(null);

type AntDConfigProviderPropsProps = Pick<AntDConfigProviderProps, 'prefixCls' | 'getPopupContainer' | 'virtual'>;

const AntDConfigProviderPropsContext = React.createContext<AntDConfigProviderPropsProps | null>(null);

/** Only to be accessed by SupportsDuBoisThemes, except for special exceptions like tests and storybook. Ask in #dubois first if you need to use it. */
export const DesignSystemThemeProvider: React.FC<DesignSystemThemeProviderProps> = ({
  isDarkMode = false,
  children,
}) => {
  return <DesignSystemThemeContext.Provider value={{ isDarkMode }}>{children}</DesignSystemThemeContext.Provider>;
};

export const DesignSystemProvider: React.FC<DesignSystemProviderProps> = ({
  children,
  enableAnimation = false,
  zIndexBase = 1000,
  getPopupContainer,
  flags = {},
  // Disable virtualization of legacy AntD components when running tests so that all items are rendered
  disableLegacyAntVirtualization = process.env.NODE_ENV === 'test' ? true : undefined,
}) => {
  const { isDarkMode } = useContext(DesignSystemThemeContext);

  const theme = useMemo(
    () =>
      getTheme(isDarkMode, {
        enableAnimation,
        zIndexBase,
      }),
    // TODO: revisit this
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isDarkMode, zIndexBase],
  );

  const providerPropsContext = useMemo<DesignSystemProviderProps>(
    () => ({
      isDarkMode,
      enableAnimation,
      zIndexBase,
      getPopupContainer,
      flags,
    }),
    [isDarkMode, enableAnimation, zIndexBase, getPopupContainer, flags],
  );

  const classNamePrefix = getClassNamePrefix(theme);

  const value = useMemo(() => {
    return {
      enableAnimation,
      isDarkMode,
      getPrefixCls: (suffix?: string) => getPrefixedClassNameFromTheme(theme, suffix),
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

  return (
    <DesignSystemProviderPropsContext.Provider value={providerPropsContext}>
      <EmotionThemeProvider theme={theme}>
        {/* This AntDConfigProviderPropsContext provides context for DesignSystemAntDConfigProvider which is used in every component */}
        <AntDConfigProviderPropsContext.Provider
          value={{ prefixCls: classNamePrefix, getPopupContainer, virtual: !disableLegacyAntVirtualization }}
        >
          <RadixTooltipProvider>
            <DesignSystemContext.Provider value={value}>{children}</DesignSystemContext.Provider>
          </RadixTooltipProvider>
        </AntDConfigProviderPropsContext.Provider>
      </EmotionThemeProvider>
    </DesignSystemProviderPropsContext.Provider>
  );
};

export const ApplyDesignSystemContextOverrides: React.FC<DesignSystemProviderProps> = ({
  enableAnimation,
  zIndexBase,
  getPopupContainer,
  flags,
  children,
}) => {
  const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
  if (parentDesignSystemProviderProps === null) {
    throw new Error(
      `ApplyDesignSystemContextOverrides cannot be used standalone - DesignSystemProvider must exist in the React context`,
    );
  }

  const newProps = useMemo(
    () => ({
      ...parentDesignSystemProviderProps,
      enableAnimation: enableAnimation ?? parentDesignSystemProviderProps.enableAnimation,
      zIndexBase: zIndexBase ?? parentDesignSystemProviderProps.zIndexBase,
      getPopupContainer: getPopupContainer ?? parentDesignSystemProviderProps.getPopupContainer,
      flags: {
        ...parentDesignSystemProviderProps.flags,
        ...flags,
      },
    }),
    [parentDesignSystemProviderProps, enableAnimation, zIndexBase, getPopupContainer, flags],
  );

  return <DesignSystemProvider {...newProps}>{children}</DesignSystemProvider>;
};

// This is a more-specific version of `ApplyDesignSystemContextOverrides` that only allows overriding the flags.
export const ApplyDesignSystemFlags: React.FC<{
  flags: DesignSystemFlags;
}> = ({ flags, children }) => {
  const parentDesignSystemProviderProps = useContext(DesignSystemProviderPropsContext);
  if (parentDesignSystemProviderProps === null) {
    throw new Error(
      `ApplyDesignSystemFlags cannot be used standalone - DesignSystemProvider must exist in the React context`,
    );
  }

  const newProps = useMemo(
    () => ({
      ...parentDesignSystemProviderProps,
      flags: {
        ...parentDesignSystemProviderProps.flags,
        ...flags,
      },
    }),
    [parentDesignSystemProviderProps, flags],
  );

  return <DesignSystemProvider {...newProps}>{children}</DesignSystemProvider>;
};

export interface DesignSystemAntDConfigProviderProps {
  children: React.ReactNode;
}

export const DesignSystemAntDConfigProvider = ({ children }: DesignSystemAntDConfigProviderProps) => {
  const antdContext = useAntDConfigProviderContext();
  return <AntDConfigProvider {...antdContext}>{children}</AntDConfigProvider>;
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
export const RestoreAntDDefaultClsPrefix = ({ children }: { children: React.ReactNode }) => {
  return <AntDConfigProvider prefixCls="ant">{children}</AntDConfigProvider>;
};
