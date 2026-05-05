import { getBottomOnlyShadowScrollStyles, useDesignSystemTheme } from '@databricks/design-system';
import type { Interpolation, Theme } from '@emotion/react';
import { useMemo } from 'react';

/**
 * Provides CSS styles for details pages (logged model details page, run details page)
 * using the unified layout style.
 */
export const useExperimentTrackingDetailsPageLayoutStyles = () => {
  const { theme } = useDesignSystemTheme();

  const detailsPageTableStyles = useMemo<Interpolation<Theme>>(
    () => ({
      minHeight: 200,
      maxHeight: 500,
      height: 'min-content',
      overflow: 'hidden',
      '& > div': {
        ...getBottomOnlyShadowScrollStyles(theme),
      },
    }),
    [theme],
  );

  const detailsPageNoEntriesStyles = useMemo<Interpolation<Theme>>(
    () => [
      {
        flex: '1',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      },
      {
        marginTop: theme.spacing.md,
      },
    ],
    [theme],
  );

  return {
    detailsPageTableStyles,
    detailsPageNoEntriesStyles,
  };
};
