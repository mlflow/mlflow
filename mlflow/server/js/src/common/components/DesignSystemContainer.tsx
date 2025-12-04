import React, { useCallback, useEffect, useRef } from 'react';
import { DesignSystemProvider, DesignSystemThemeProvider } from '@databricks/design-system';
import { ColorsPaletteDatalist } from './ColorsPaletteDatalist';
import { Theme } from '@emotion/react';
import { PATTERN_FLY_TOKEN_TRANSLATION } from '../styles/patternfly/patternflyTokenTranslation';
import { ThemeProvider as EmotionThemeProvider } from '@emotion/react';
import '../styles/patternfly/pf-shell-overrides.scss';

const isInsideShadowDOM = (element: HTMLDivElement | null): boolean =>
  element instanceof window.Node && element.getRootNode() !== document;

type DesignSystemContainerProps = {
  isDarkTheme?: boolean;
  children: React.ReactNode;
};

const ThemeProvider = ({ children, isDarkTheme }: { children?: React.ReactNode; isDarkTheme?: boolean }) => {
  // eslint-disable-next-line react/forbid-elements
  return <DesignSystemThemeProvider isDarkMode={isDarkTheme}>{children}</DesignSystemThemeProvider>;
};

export const MLflowImagePreviewContainer = React.createContext({
  getImagePreviewPopupContainer: () => document.body,
});

/**
 * MFE-safe DesignSystemProvider that checks if the application is
 * in the context of the Shadow DOM and if true, provides dedicated
 * DOM element for the purpose of housing modals/popups there.
 */
export const DesignSystemContainer = (props: DesignSystemContainerProps) => {
  const modalContainerElement = useRef<HTMLDivElement | null>(null);
  const { isDarkTheme = false, children } = props;

  const getPopupContainer = useCallback(() => {
    const modelContainerEle = modalContainerElement.current;
    if (modelContainerEle !== null && isInsideShadowDOM(modelContainerEle)) {
      return modelContainerEle;
    }
    return document.body;
  }, []);

  // Specialized container for antd image previews, always rendered near MLflow
  // to maintain prefixed CSS classes and styles.
  const getImagePreviewPopupContainer = useCallback(() => {
    const modelContainerEle = modalContainerElement.current;
    if (modelContainerEle !== null) {
      return modelContainerEle;
    }
    return document.body;
  }, []);

  useEffect(() => {
    const patternflyDarkModeSwitcher = document.getElementById('patternfly-dark-mode-switcher');
    if (patternflyDarkModeSwitcher) {
      if (isDarkTheme) {
        patternflyDarkModeSwitcher.classList.add('pf-v6-theme-dark');
      } else {
        patternflyDarkModeSwitcher.classList.remove('pf-v6-theme-dark');
      }
    }
  }, [isDarkTheme]);

  return (
    <ThemeProvider isDarkTheme={isDarkTheme}>
      <DesignSystemProvider getPopupContainer={getPopupContainer} {...props}>
        <MLflowImagePreviewContainer.Provider value={{ getImagePreviewPopupContainer }}>
          <EmotionThemeProvider theme={(baseTheme) => PATTERN_FLY_TOKEN_TRANSLATION(baseTheme)}>
            <div className="pf-shell-container">{children}</div>
            <div ref={modalContainerElement} />
          </EmotionThemeProvider>
        </MLflowImagePreviewContainer.Provider>
      </DesignSystemProvider>
      <ColorsPaletteDatalist />
    </ThemeProvider>
  );
};
