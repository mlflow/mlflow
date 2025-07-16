import { isNil } from 'lodash';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { SearchMatch } from './ModelTrace.types';
import { ModelTraceExplorerHighlightedCodeSnippet } from './ModelTraceExplorerHighlightedCodeSnippet';
import { ModelTraceExplorerHighlightedSnippetTitle } from './ModelTraceExplorerHighlightedSnippetTitle';
// eslint-disable-next-line import/no-deprecated
import { CodeSnippet } from '../snippet';

export function ModelTraceExplorerAttributeRow({
  title,
  value,
  searchFilter,
  activeMatch,
  containsActiveMatch,
}: {
  title: string;
  // values can be arbitrary JSON
  value: string;
  searchFilter: string;
  // the current active search match
  activeMatch: SearchMatch | null;
  // whether the attribute row being rendered contains the
  // current active match (either in the title or value)
  containsActiveMatch: boolean;
}) {
  const stringValue = useMemo(() => JSON.stringify(value, null, 2), [value]);
  const containsMatches =
    Boolean(searchFilter) && !isNil(activeMatch) && stringValue.toLowerCase().includes(searchFilter);
  const [expanded, setExpanded] = useState(containsMatches);
  const [isContentLong, setIsContentLong] = useState(false);
  const snippetRef = useRef<HTMLDivElement>(null);

  // if the content is not expanded, render it as a single line that will get truncated
  const displayValue = useMemo(() => (expanded ? stringValue : JSON.stringify(value)), [value, stringValue, expanded]);
  const { theme } = useDesignSystemTheme();

  const PreWithRef = useCallback((preProps: any) => <pre {...preProps} ref={snippetRef} />, []);
  const isTitleMatch = containsActiveMatch && (activeMatch?.isKeyMatch ?? false);

  useEffect(() => {
    if (snippetRef.current) {
      setIsContentLong(snippetRef.current.scrollWidth > snippetRef.current.clientWidth);
    }
  }, [value]);

  // the returned fragment must have 3 children
  // because the parent is a grid with 3 columns
  return (
    <>
      <Typography.Text
        color="secondary"
        css={{
          display: 'inline-block',
          wordBreak: 'break-all',
        }}
        title={title}
      >
        <ModelTraceExplorerHighlightedSnippetTitle
          title={title}
          searchFilter={searchFilter}
          isActiveMatch={isTitleMatch}
        />
      </Typography.Text>
      {containsMatches ? (
        <>
          <div />
          <ModelTraceExplorerHighlightedCodeSnippet
            searchFilter={searchFilter}
            data={stringValue}
            activeMatch={activeMatch}
            containsActiveMatch={containsActiveMatch}
          />
        </>
      ) : (
        <>
          {isContentLong ? (
            <Button
              size="small"
              componentId={
                expanded
                  ? 'shared.model-trace-explorer.collapse-attribute-row'
                  : 'shared.model-trace-explorer.expand-attribute-row'
              }
              type="tertiary"
              icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
              onClick={() => setExpanded(!expanded)}
            />
          ) : (
            <div />
          )}
          {/* eslint-disable-next-line import/no-deprecated */}
          <CodeSnippet
            PreTag={PreWithRef}
            language="json"
            lineProps={{
              style: {
                wordBreak: 'break-all',
                whiteSpace: 'pre-wrap',
              },
            }}
            wrapLines={expanded}
            style={{
              backgroundColor: theme.colors.backgroundSecondary,
              padding: theme.spacing.xs,
              overflow: expanded ? 'auto' : 'hidden',
              textOverflow: expanded ? 'unset' : 'ellipsis',
              height: 'fit-content',
            }}
          >
            {displayValue}
          </CodeSnippet>
        </>
      )}
    </>
  );
}
