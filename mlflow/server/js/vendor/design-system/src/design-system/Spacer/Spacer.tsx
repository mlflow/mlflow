import { css } from '@emotion/react';

import { useDesignSystemTheme } from '../Hooks';
import type { HTMLDataAttributes } from '../types';

type validSpacerOptions = 'xs' | 'sm' | 'md' | 'lg';

export interface SpacerProps extends HTMLDataAttributes {
  size?: validSpacerOptions;
  /** Prevents the Spacer component from shrinking when used in flexbox columns. **/
  shrinks?: boolean;
}

export const Spacer: React.FC<SpacerProps> = ({ size = 'md', shrinks, ...props }) => {
  const { theme } = useDesignSystemTheme();

  const spacingValues: Record<validSpacerOptions, number> = {
    xs: theme.spacing.xs,
    sm: theme.spacing.sm,
    md: theme.spacing.md,
    lg: theme.spacing.lg,
  };

  return (
    <div
      css={css({
        height: spacingValues[size],
        ...(shrinks === false ? { flexShrink: 0 } : undefined),
      })}
      {...props}
    />
  );
};
