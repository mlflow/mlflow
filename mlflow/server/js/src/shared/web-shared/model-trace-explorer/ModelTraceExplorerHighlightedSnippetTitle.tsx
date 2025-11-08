import React, { useCallback } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

import { ACTIVE_HIGHLIGHT_COLOR, INACTIVE_HIGHLIGHT_COLOR } from './constants';

export const ModelTraceExplorerHighlightedSnippetTitle = ({
  title,
  searchFilter,
  isActiveMatch,
}: {
  title: string;
  searchFilter: string;
  isActiveMatch: boolean;
}): React.ReactElement => {
  const { theme } = useDesignSystemTheme();
  const scrollToActiveMatch = useCallback((node: HTMLElement | null) => {
    node?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, []);

  const titleLower = title.toLowerCase();
  if (!titleLower.includes(searchFilter)) {
    return <div>{title}</div>;
  }

  const startIdx = titleLower.indexOf(searchFilter);
  const endIdx = startIdx + searchFilter.length;
  const backgroundColor = isActiveMatch ? theme.colors[ACTIVE_HIGHLIGHT_COLOR] : theme.colors[INACTIVE_HIGHLIGHT_COLOR];

  return (
    <div>
      {title.slice(0, startIdx)}
      <span ref={isActiveMatch ? scrollToActiveMatch : null} css={{ backgroundColor, scrollMarginTop: 50 }}>
        {title.slice(startIdx, endIdx)}
      </span>
      {title.slice(endIdx)}
    </div>
  );
};
