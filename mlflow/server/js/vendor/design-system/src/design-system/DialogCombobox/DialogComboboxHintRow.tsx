import React from 'react';

import { useDesignSystemTheme } from '../Hooks';

interface DialogComboboxHintRowProps {
  children: React.ReactNode;
}

export const DialogComboboxHintRow = ({ children }: DialogComboboxHintRowProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        minWidth: '100%',
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
        '[data-disabled] &': {
          color: theme.colors.actionDisabledText,
        },
      }}
    >
      {children}
    </div>
  );
};
