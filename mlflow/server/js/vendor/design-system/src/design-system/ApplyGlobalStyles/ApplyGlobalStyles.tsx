import { Global, css, type SerializedStyles } from '@emotion/react';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';

export const getGlobalStyles = (theme: Theme): SerializedStyles => {
  return css({
    'body, .mfe-root': {
      backgroundColor: theme.colors.backgroundPrimary,
      color: theme.colors.textPrimary,
      '--dubois-global-background-color': theme.colors.backgroundPrimary,
      '--dubois-global-color': theme.colors.textPrimary,
    },
  });
};

export const ApplyGlobalStyles: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  return <Global styles={getGlobalStyles(theme)} />;
};
