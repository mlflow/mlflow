import { css } from '@emotion/react';
import type { SerializedStyles } from '@emotion/react';
import React from 'react';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronDownIcon } from '../Icon';

export interface DownshiftToggleButtonProps {
  id: string;
  onClick: (e: React.SyntheticEvent) => void;
  tabIndex: number;
}

interface TypeaheadComboboxToggleButtonProps extends DownshiftToggleButtonProps {
  disabled?: boolean;
}

const getToggleButtonStyles = (theme: Theme, disabled?: boolean): SerializedStyles => {
  return css({
    cursor: 'pointer',
    userSelect: 'none',
    color: theme.colors.textSecondary,
    backgroundColor: 'transparent',
    border: 'none',
    padding: 0,
    marginLeft: theme.spacing.xs,
    height: 16,
    width: 16,

    ...(disabled && {
      pointerEvents: 'none',
      color: theme.colors.actionDisabledText,
    }),
  });
};

export const TypeaheadComboboxToggleButton = React.forwardRef<HTMLButtonElement, TypeaheadComboboxToggleButtonProps>(
  ({ disabled, ...restProps }, ref) => {
    const { theme } = useDesignSystemTheme();
    const { onClick } = restProps;

    function handleClick(e: React.SyntheticEvent) {
      e.stopPropagation();
      onClick(e);
    }

    return (
      <button
        type="button"
        aria-label="toggle menu"
        ref={ref}
        css={getToggleButtonStyles(theme, disabled)}
        {...restProps}
        onClick={handleClick}
      >
        <ChevronDownIcon />
      </button>
    );
  },
);
