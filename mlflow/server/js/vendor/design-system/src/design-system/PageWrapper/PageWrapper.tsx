import { css } from '@emotion/react';

import { useDesignSystemTheme } from '../Hooks';
import type { HTMLDataAttributes } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface PageWrapperProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {}

export const PageWrapper: React.FC<PageWrapperProps> = ({ children, ...props }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      {...addDebugOutlineIfEnabled()}
      css={css({
        paddingLeft: 16,
        paddingRight: 16,
        backgroundColor: theme.isDarkMode ? theme.colors.backgroundPrimary : 'transparent',
      })}
      {...props}
    >
      {children}
    </div>
  );
};
