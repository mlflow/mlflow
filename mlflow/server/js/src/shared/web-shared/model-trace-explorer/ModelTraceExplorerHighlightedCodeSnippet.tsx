import React, { useCallback, useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

import type { SearchMatch } from './ModelTrace.types';
import { getHighlightedSpanComponents } from './ModelTraceExplorer.utils';
import { ACTIVE_HIGHLIGHT_COLOR, INACTIVE_HIGHLIGHT_COLOR } from './constants';

export const ModelTraceExplorerHighlightedCodeSnippet = ({
  searchFilter,
  data,
  activeMatch,
  containsActiveMatch,
}: {
  searchFilter: string;
  data: string;
  activeMatch: SearchMatch;
  containsActiveMatch: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const scrollToActiveMatch = useCallback((node: HTMLElement | null) => {
    node?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, []);

  const spans = useMemo(() => {
    if (!searchFilter) {
      return [];
    }

    return getHighlightedSpanComponents({
      data,
      searchFilter,
      activeMatchBackgroundColor: theme.colors[ACTIVE_HIGHLIGHT_COLOR],
      inactiveMatchBackgroundColor: theme.colors[INACTIVE_HIGHLIGHT_COLOR],
      containsActiveMatch,
      activeMatch,
      scrollToActiveMatch,
    });
  }, [searchFilter, data, theme, containsActiveMatch, activeMatch, scrollToActiveMatch]);

  return (
    <pre
      css={{
        whiteSpace: 'pre-wrap',
        backgroundColor: theme.colors.backgroundSecondary,
        padding: theme.spacing.sm,
        fontSize: theme.typography.fontSizeSm,
      }}
    >
      {spans}
    </pre>
  );
};
