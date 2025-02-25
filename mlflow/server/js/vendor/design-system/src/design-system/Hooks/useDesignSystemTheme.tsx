import { useTheme as useEmotionTheme } from '@emotion/react';
import type { ReactElement } from 'react';
import React from 'react';

import type { Theme } from '../../theme';
import { getTheme } from '../../theme';

export interface DesignSystemThemeInterface {
  /** Theme object that contains colors, spacing, and other variables **/
  theme: Theme;
  /** Prefix that is used in front of each className.  **/
  classNamePrefix: string;
  /** Helper method that use be used to construct the full className of an underlying AntD className.
   * Use with caution and prefer emotion.js when possible **/
  getPrefixedClassName: (className: string) => string;
}

export function getClassNamePrefix(theme: Theme): string {
  const antdThemeName = theme.isDarkMode ? 'dark' : 'light';
  return `${theme.general.classnamePrefix}-${antdThemeName}`;
}

export function getPrefixedClassNameFromTheme(theme: Theme, className: string | null | undefined): string {
  return [getClassNamePrefix(theme), className].filter(Boolean).join('-');
}

export function useDesignSystemTheme(): DesignSystemThemeInterface {
  const emotionTheme = useEmotionTheme() as Theme;
  // Graceful fallback to default theme in case a test or developer forgot context.
  const theme = emotionTheme && emotionTheme.general ? emotionTheme : getTheme(false);

  return {
    theme: theme,
    classNamePrefix: getClassNamePrefix(theme),
    getPrefixedClassName: (className: string) => getPrefixedClassNameFromTheme(theme, className),
  };
}

export type DesignSystemHocProps = { designSystemThemeApi: DesignSystemThemeInterface };

// This is a simple typed HOC wrapper around the useDesignSystemTheme hook, for use in older react components.
export function WithDesignSystemThemeHoc<P>(Component: React.ComponentType<P & DesignSystemHocProps>) {
  return function WrappedWithDesignSystemTheme(props: P): ReactElement {
    const themeValues = useDesignSystemTheme();

    return <Component {...props} designSystemThemeApi={themeValues} />;
  };
}
