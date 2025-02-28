import type { ThemeType } from '@databricks/design-system';

import { css } from '@emotion/react';

import { safex } from './safex';

export const getDebugOutlineStyles = (theme: ThemeType) =>
  css({
    outline: `1px dashed ${theme.isDarkMode ? theme.colors.lime : theme.colors.lime}`,
    outlineOffset: '2px',
  });

export function addDebugOutlineIfEnabled() {
  return safex('databricks.fe.designsystem.showDebugOutline', false) ? { 'data-dubois-show-outline': true } : {};
}

export function addDebugOutlineStylesIfEnabled(theme: ThemeType) {
  return safex('databricks.fe.designsystem.showDebugOutline', false) ? getDebugOutlineStyles(theme) : {};
}
