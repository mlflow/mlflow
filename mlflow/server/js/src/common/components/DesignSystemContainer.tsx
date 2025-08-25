import React, { useCallback, useRef } from 'react';
import { DesignSystemProvider, DesignSystemThemeProvider } from '@databricks/design-system';
import { ColorsPaletteDatalist } from './ColorsPaletteDatalist';

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

  return (
    <ThemeProvider isDarkTheme={isDarkTheme}>
      <DesignSystemProvider getPopupContainer={getPopupContainer} {...props}>
        <MLflowImagePreviewContainer.Provider value={{ getImagePreviewPopupContainer }}>
          {children}
          <div ref={modalContainerElement} />
        </MLflowImagePreviewContainer.Provider>
      </DesignSystemProvider>
      <ColorsPaletteDatalist />
    </ThemeProvider>
  );
};
