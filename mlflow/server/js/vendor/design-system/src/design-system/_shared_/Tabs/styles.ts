import type { CSSObject } from '@emotion/react';

import type { Theme } from '../../../theme';

export const getCommonTabsListStyles = (theme: Theme): CSSObject => {
  return {
    display: 'flex',
    borderBottom: `1px solid ${theme.colors.border}`,
    marginBottom: theme.spacing.md,
    height: theme.general.heightSm,
    boxSizing: 'border-box',
  };
};

export const getCommonTabsTriggerStyles = (theme: Theme): CSSObject => {
  return {
    display: 'flex',
    fontWeight: theme.typography.typographyBoldFontWeight,
    fontSize: theme.typography.fontSizeMd,
    backgroundColor: 'transparent',
    marginRight: theme.spacing.md,
  };
};
