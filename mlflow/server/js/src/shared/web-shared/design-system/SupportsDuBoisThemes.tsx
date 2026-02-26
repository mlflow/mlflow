import React from 'react';

import { DesignSystemThemeProvider } from '@databricks/design-system';

export interface SupportsDuBoisThemesProps {
  disabled?: boolean;
}

export const SupportsDuBoisThemes: React.FC<React.PropsWithChildren<SupportsDuBoisThemesProps>> = ({
  disabled = false,
  children,
}) => {
  // eslint-disable-next-line react/forbid-elements
  return <DesignSystemThemeProvider isDarkMode={false}>{children}</DesignSystemThemeProvider>;
};
