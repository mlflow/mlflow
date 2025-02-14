import React from 'react';

import { getSelectedItemStyles } from './TypeaheadComboboxSelectedItem';
import { useDesignSystemTheme } from '../Hooks';

export interface CountBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  countStartAt?: number;
  totalCount: number;
  disabled?: boolean;
}

export const CountBadge: React.FC<CountBadgeProps> = ({ countStartAt, totalCount, disabled }: CountBadgeProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={[
        getSelectedItemStyles(theme),
        { paddingInlineEnd: theme.spacing.xs, ...(disabled && { color: theme.colors.actionDisabledText }) },
      ]}
    >
      {countStartAt ? `+${totalCount - countStartAt}` : totalCount}
    </div>
  );
};
