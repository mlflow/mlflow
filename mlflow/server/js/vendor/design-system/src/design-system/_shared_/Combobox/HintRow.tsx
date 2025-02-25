import React from 'react';

import { useDesignSystemTheme } from '../../Hooks';

interface HintRowProps {
  disabled?: boolean;
  children: React.ReactNode;
}

export const HintRow = ({ disabled, children }: HintRowProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
        ...(disabled && {
          color: theme.colors.actionDisabledText,
        }),
      }}
    >
      {children}
    </div>
  );
};
