import { getBottomOnlyShadowScrollStyles, useDesignSystemTheme } from '@databricks/design-system';
import type { Interpolation, Theme } from '@emotion/react';
import { useMemo } from 'react';

/**
 * Provides CSS styles for details pages (logged model details page, run details page)
 * depending on currently enabled layout, based on the feature flag.
 */
export const useExperimentTrackingDetailsPageLayoutStyles = () => {
  const { theme } = useDesignSystemTheme();
  const usingUnifiedDetailsLayout = false;

  const detailsPageTableStyles = useMemo<Interpolation<Theme>>(
    () =>
      usingUnifiedDetailsLayout
        ? {
            height: 200,
            overflow: 'hidden',
            '& > div': {
              ...getBottomOnlyShadowScrollStyles(theme),
            },
          }
        : {},
    [theme, usingUnifiedDetailsLayout],
  );

  const detailsPageNoResultsWrapperStyles = useMemo<Interpolation<Theme>>(
    () => (usingUnifiedDetailsLayout ? {} : { marginTop: theme.spacing.md * 4 }),
    [theme, usingUnifiedDetailsLayout],
  );

  const detailsPageNoEntriesStyles = useMemo<Interpolation<Theme>>(
    () => [
      {
        flex: '1',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      },
      usingUnifiedDetailsLayout && {
        marginTop: theme.spacing.md,
      },
    ],
    [theme, usingUnifiedDetailsLayout],
  );

  return {
    usingUnifiedDetailsLayout,
    detailsPageTableStyles,
    detailsPageNoEntriesStyles,
    detailsPageNoResultsWrapperStyles,
  };
};
