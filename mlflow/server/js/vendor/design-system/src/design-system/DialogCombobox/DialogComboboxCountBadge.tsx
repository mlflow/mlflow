import type { Theme, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDesignSystemTheme } from '../Hooks';
import { importantify } from '../utils/css-utils';

const getCountBadgeStyles = (theme: Theme): SerializedStyles =>
  css(
    importantify({
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',

      boxSizing: 'border-box',
      padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
      background: theme.colors.tagDefault,
      borderRadius: theme.general.borderRadiusBase,

      fontSize: theme.typography.fontSizeBase,
      height: 20,
    }),
  );

export const DialogComboboxCountBadge = (props: { countStartAt?: number } & any) => {
  const { countStartAt, ...restOfProps } = props;
  const { theme } = useDesignSystemTheme();
  const { value } = useDialogComboboxContext();

  return (
    <div {...restOfProps} css={getCountBadgeStyles(theme)}>
      {Array.isArray(value) ? (countStartAt ? `+${value.length - countStartAt}` : value.length) : value ? 1 : 0}
    </div>
  );
};
