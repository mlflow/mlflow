import React, { useState } from 'react';
import { useDebouncedCallback } from 'use-debounce';

import {
  Button,
  ChevronDownIcon,
  ChevronUpIcon,
  Input,
  SearchIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import type { SearchMatch } from './ModelTrace.types';

const ModelTraceExplorerSearchBox = ({
  searchFilter,
  setSearchFilter,
  matchData,
  handleNextSearchMatch,
  handlePreviousSearchMatch,
}: {
  searchFilter: string;
  setSearchFilter: (searchFilter: string) => void;
  matchData: {
    match: SearchMatch | null;
    totalMatches: number;
    currentMatchIndex: number;
  };
  handleNextSearchMatch: () => void;
  handlePreviousSearchMatch: () => void;
}) => {
  const [searchValue, setSearchValue] = useState(searchFilter);
  const debouncedSetSearchFilter = useDebouncedCallback(setSearchFilter, 350);
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between',
        gap: theme.spacing.sm,
      }}
    >
      <Input
        componentId="shared.model-trace-explorer.search-input"
        allowClear
        placeholder="Search"
        value={searchValue}
        onClear={() => {
          setSearchFilter('');
          setSearchValue('');
        }}
        onChange={(e) => {
          setSearchValue(e.target.value);
          debouncedSetSearchFilter(e.target.value.toLowerCase());
        }}
        prefix={<SearchIcon />}
        css={{
          width: '100%',
          boxSizing: 'border-box',
        }}
      />
      {matchData.match && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            marginLeft: theme.spacing.xs,
            marginRight: theme.spacing.sm,
            alignItems: 'center',
          }}
        >
          <Typography.Text css={{ whiteSpace: 'nowrap', marginRight: theme.spacing.sm }}>
            {matchData.currentMatchIndex + 1} / {matchData.totalMatches}
          </Typography.Text>
          <Button
            data-testid="prev-search-match"
            icon={<ChevronUpIcon />}
            onClick={handlePreviousSearchMatch}
            componentId="shared.model-trace-explorer.prev-search-match"
          />
          <Button
            data-testid="next-search-match"
            icon={<ChevronDownIcon />}
            onClick={handleNextSearchMatch}
            componentId="shared.model-trace-explorer.next-search-match"
          />
        </div>
      )}
    </div>
  );
};

export default ModelTraceExplorerSearchBox;
